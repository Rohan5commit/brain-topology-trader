#!/usr/bin/env python3
"""NCP v5 seed 1 — cross-sectional attention — OVH H100 cuda:0."""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
SEED = 1
DEVICE_ID = 0

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ncps", "yfinance"], check=True)

import torch as _t
assert _t.cuda.is_available(), "No CUDA"
print(f"GPU {DEVICE_ID}: {_t.cuda.get_device_name(DEVICE_ID)}")
del _t

import logging, random
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger(__name__)

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED); random.seed(SEED)
log.info("Seed: %d | device: cuda:%d", SEED, DEVICE_ID)

import config
from data.features import FeatureEngineer, FEATURE_NAMES
from model.ncp_model_v5 import NCPTradingModelV5
import yfinance as yf

WORK = "/workspace"
WEIGHTS_PATH = f"{WORK}/ncp_v5_seed1.pt"
CKPT_TXT     = f"{WORK}/checkpoint_v5_seed1.txt"
os.makedirs(f"{WORK}/ckpts_v5_seed1", exist_ok=True)

EPOCHS = 60
BATCH  = 4096

start_epoch = 0
if os.path.exists(CKPT_TXT) and os.path.exists(WEIGHTS_PATH):
    with open(CKPT_TXT) as f:
        start_epoch = int(f.read().strip())
    log.info("Resuming from epoch %d", start_epoch)

def _ckpt(model, epoch):
    torch.save(model.state_dict(), WEIGHTS_PATH)
    torch.save(model.state_dict(), f"{WORK}/ckpts_v5_seed1/epoch_{epoch:03d}.pt")
    with open(CKPT_TXT, "w") as f: f.write(str(epoch))
    log.info("Checkpoint saved epoch %d", epoch)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__(); self.gamma = gamma; self.weight = weight
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

device = torch.device(f"cuda:{DEVICE_ID}")
TICKERS = config.TICKER_UNIVERSE
ticker_to_idx = {t: i for i, t in enumerate(TICKERS)}

log.info("Fetching macro…")
macro_df = spy_5d_fwd = None
try:
    macro_raw = yf.download(["^VIX","^TNX","^IRX","SPY"], start="2000-01-01", end="2025-12-31",
        interval="1d", auto_adjust=True, progress=False)
    mc = macro_raw["Close"] if isinstance(macro_raw.columns, pd.MultiIndex) else macro_raw
    mc.index = pd.to_datetime(mc.index).normalize()
    mc = mc.rename(columns={"^VIX": "vix"})
    mc["yield_curve_slope"] = mc["^TNX"] - mc["^IRX"]
    mc["spy_1d_return"] = mc["SPY"].pct_change(1)
    spy_5d_fwd = mc["SPY"].pct_change(5).shift(-5)
    macro_df = mc[["vix","yield_curve_slope","spy_1d_return"]].ffill().fillna(0.0)
    log.info("Macro: %d rows", len(macro_df))
except Exception as e:
    log.warning("Macro failed: %s", e)

spy_3d_cum = macro_df["spy_1d_return"].rolling(3).sum().fillna(0.0) if macro_df is not None else None

log.info("Pass 1: features…")
engineer = FeatureEngineer()
all_feat_dfs, all_close = {}, {}

for bs in range(0, len(TICKERS), 100):
    batch = TICKERS[bs:bs+100]
    log.info("Batch %d–%d", bs+1, bs+len(batch))
    try:
        raw = yf.download(batch, start="2000-01-01", end="2025-12-31",
            interval="1d", auto_adjust=True, group_by="ticker", progress=False)
    except Exception as e:
        log.warning("Batch fail: %s", e); continue
    for ticker in batch:
        try:
            if len(batch)==1: df=raw.copy()
            elif isinstance(raw.columns, pd.MultiIndex):
                if ticker in raw.columns.get_level_values(0): df=raw[ticker].copy()
                elif ticker in raw.columns.get_level_values(1): df=raw.xs(ticker,level=1,axis=1).copy()
                else: continue
            else: continue
            df=df.rename(columns=str.lower).dropna(how="all")
            df.index=pd.to_datetime(df.index).normalize()
            if len(df)<config.SEQUENCE_LENGTH+config.RETURN_HORIZON+5: continue
            feat_df=engineer._stock_features(df)
            if feat_df is None or len(feat_df)<config.SEQUENCE_LENGTH+config.RETURN_HORIZON: continue
            if macro_df is not None:
                aligned=macro_df.reindex(feat_df.index,method="ffill")
                feat_df["vix"]=aligned["vix"].fillna(20.0).values
                feat_df["yield_curve_slope"]=aligned["yield_curve_slope"].fillna(0.5).values
                feat_df["spy_1d_return"]=aligned["spy_1d_return"].fillna(0.0).values
            else:
                feat_df["vix"]=20.0; feat_df["yield_curve_slope"]=0.5; feat_df["spy_1d_return"]=0.0
            all_feat_dfs[ticker]=feat_df; all_close[ticker]=df["close"]
        except Exception as e:
            log.warning("Skip %s: %s", ticker, e)
log.info("Features: %d tickers", len(all_feat_dfs))

returns_panel=pd.DataFrame({t: fd["returns_20d"] for t,fd in all_feat_dfs.items()})
rank_panel=returns_panel.rank(axis=1,pct=True).fillna(0.5)
fwd_excess={}
for ticker,feat_df in all_feat_dfs.items():
    close=all_close[ticker].reindex(feat_df.index)
    stock_5d=close.pct_change(5).shift(-5)
    fwd_excess[ticker]=(stock_5d - spy_5d_fwd.reindex(feat_df.index,method="ffill").fillna(0.0)
                        if spy_5d_fwd is not None else stock_5d)
excess_df=pd.DataFrame(fwd_excess)
q_low=excess_df.quantile(config.QUARTILE_THRESHOLD,axis=1)
q_high=excess_df.quantile(1-config.QUARTILE_THRESHOLD,axis=1)

log.info("Pass 2: building samples (fp16)…")
all_X,all_y,all_idx,all_sector=[],[],[],[]
for ticker,feat_df in all_feat_dfs.items():
    try:
        feat_df=feat_df.copy()
        feat_df["momentum_rank"]=(rank_panel[ticker].reindex(feat_df.index).ffill().fillna(0.5).values
            if ticker in rank_panel.columns else 0.5)
        stock_3d=feat_df["returns_1d"].rolling(3).sum().fillna(0.0)
        abnormal=(stock_3d-spy_3d_cum.reindex(feat_df.index,method="ffill").fillna(0.0)
                  if spy_3d_cum is not None else stock_3d)
        feat_df["sentiment_3d"]=abnormal.clip(-0.15,0.15)/0.15
        feat_arr=feat_df[FEATURE_NAMES].values.astype(np.float16)
        feat_arr=np.nan_to_num(feat_arr,nan=0.0,posinf=0.0,neginf=0.0)
        ticker_idx=ticker_to_idx.get(ticker,0)
        sector_idx=config.TICKER_SECTOR.get(ticker,12)
        ticker_excess=fwd_excess.get(ticker)
        for i in range(config.SEQUENCE_LENGTH,len(feat_arr)-config.RETURN_HORIZON):
            if ticker_excess is None: continue
            date=feat_df.index[i]
            if date not in q_low.index: continue
            exc_ret=ticker_excess.iloc[i] if i<len(ticker_excess) else np.nan
            if pd.isna(exc_ret): continue
            ql,qh=q_low.loc[date],q_high.loc[date]
            if pd.isna(ql) or pd.isna(qh): continue
            if exc_ret>=qh: label=1
            elif exc_ret<=ql: label=0
            else: continue
            all_X.append(feat_arr[i-config.SEQUENCE_LENGTH:i])
            all_y.append(label); all_idx.append(ticker_idx); all_sector.append(sector_idx)
    except Exception as e:
        log.warning("Skip %s p2: %s", ticker, e)

log.info("Samples: %d", len(all_X))
y_np=np.array(all_y)
counts=np.bincount(y_np,minlength=2).astype(np.float32)
weights=1.0/counts.clip(min=1); weights=weights/weights.mean()
log.info("Class weights: %.3f / %.3f", *weights)

X=torch.HalfTensor(np.array(all_X)).to(device)
y=torch.LongTensor(all_y).to(device)
idx_t=torch.LongTensor(all_idx).to(device)
sec_t=torch.LongTensor(all_sector).to(device)
log.info("X on %s: %.1f GB fp16", device, X.element_size()*X.nelement()/1e9)
used=torch.cuda.memory_allocated(DEVICE_ID)/1e9
total=torch.cuda.get_device_properties(DEVICE_ID).total_memory/1e9
log.info("GPU mem: %.1f / %.1f GB", used, total)

dataset=TensorDataset(X,idx_t,sec_t,y)
loader=DataLoader(dataset,batch_size=BATCH,shuffle=True,num_workers=0)

model=NCPTradingModelV5(
    num_stocks=len(TICKERS), num_features=config.NUM_FEATURES,
    input_size=config.INPUT_SIZE, ncp_units=config.NCP_UNITS,
    ncp_output_size=config.NCP_OUTPUT_SIZE, ncp_sparsity=config.NCP_SPARSITY,
    embedding_dim=config.EMBEDDING_DIM, num_sectors=config.NUM_SECTORS,
    sector_embedding_dim=config.SECTOR_EMBEDDING_DIM,
    cs_heads=4, cs_dropout=0.1, dropout=config.DROPOUT,
).to(device)

if start_epoch>0:
    model.load_state_dict(torch.load(WEIGHTS_PATH,map_location=device))
    log.info("Loaded checkpoint epoch %d", start_epoch)

optimizer=torch.optim.AdamW(model.parameters(),lr=config.LEARNING_RATE,weight_decay=config.WEIGHT_DECAY)
for pg in optimizer.param_groups: pg["initial_lr"]=config.LEARNING_RATE
scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,T_0=config.SGDR_T0,T_mult=config.SGDR_T_MULT,
    last_epoch=start_epoch-1 if start_epoch>0 else -1)
criterion=FocalLoss(gamma=config.FOCAL_GAMMA,weight=torch.FloatTensor(weights).to(device))

log.info("Training epochs %d–%d", start_epoch+1, EPOCHS)
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss,correct,n=0.0,0,0
    for xb,ib,sb,yb in loader:
        xb=xb.float()
        optimizer.zero_grad()
        logits=model(xb,ib,sb)
        loss=criterion(logits,yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        total_loss+=loss.item()*len(yb); correct+=(logits.argmax(1)==yb).sum().item(); n+=len(yb)
    scheduler.step()
    log.info("Epoch %d/%d | loss=%.4f | acc=%.4f | lr=%.2e",
        epoch+1,EPOCHS,total_loss/n,correct/n,scheduler.get_last_lr()[0])
    _ckpt(model,epoch+1)

log.info("Done. Weights: %s", WEIGHTS_PATH)
print(f"\n=== NCP V5 SEED {SEED} COMPLETE — {WEIGHTS_PATH} ===")
