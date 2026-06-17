# Model hyperparameters
NCP_UNITS = 64
NCP_OUTPUT_SIZE = 3          # 0=buy, 1=hold, 2=sell
NCP_SPARSITY = 0.5
EMBEDDING_DIM = 16
SEQUENCE_LENGTH = 20         # days of history fed to LTC
NUM_FEATURES = 17            # features per stock per day
INPUT_SIZE = NUM_FEATURES + EMBEDDING_DIM  # 33

# Training
LEARNING_RATE = 1e-4
BATCH_SIZE = 512
HISTORICAL_EPOCHS = 50
HISTORICAL_START = "2000-01-01"
HISTORICAL_END = "2024-12-31"

# Signal
SIGNAL_THRESHOLD = 0.55
SIGNAL_SMOOTH_DAYS = 3

# Position management
MAX_POSITION_PCT = 0.05      # 5% of portfolio per stock
KELLY_B = 1.5                # avg_win / avg_loss ratio
MIN_HOLD_DAYS = 3
PORTFOLIO_VALUE = 200_000    # $200K effective (2x on $100K margin)

# Reward weights
ALPHA = 0.5                  # direction accuracy weight
BETA = 0.5                   # pnl_return weight

# Twelve Data
KEY_ROTATION_COUNT = 8
TWELVEDATA_INTERVAL = "1day"
FETCH_BATCH_SIZE = 8         # tickers per API call (outputsize param)

# Paths (Modal Volume /data)
DATA_DIR = "/data"
WEIGHTS_LATEST_PATH = "/data/ncp_weights_latest.pt"
WEIGHTS_BASE_PATH = "/data/ncp_weights_base.pt"
SIGNALS_PATH = "/data/signals_history.parquet"
POSITIONS_PATH = "/data/positions_state.parquet"
FEATURES_CACHE_PATH = "/data/features_cache.parquet"

# Alpaca
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# Top ~800 US stocks by market cap (S&P 500 + Russell 1000, deduped, sorted approx by mkt cap)
TICKER_UNIVERSE = [
    # ── Mega-cap Tech ──────────────────────────────────────────────────────────
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO",
    "ORCL", "CRM", "AMD", "ADBE", "INTU", "QCOM", "TXN", "AMAT", "LRCX",
    "KLAC", "MRVL", "SNPS", "CDNS", "FTNT", "PANW", "CRWD", "NOW", "WDAY",
    "DDOG", "SNOW", "MU", "INTC", "IBM", "CSCO", "ACN", "HPQ", "DELL",
    "CTSH", "EPAM", "NET", "TWLO", "VEEV", "HUBS", "ANSS", "PAYC", "ZM",
    "DOCU", "OKTA", "ZS", "SPLK", "DOMO", "JAMF", "BRZE", "MNDY", "FRSH",
    "GTLB", "APPN", "FIVN", "NICE", "CSGP", "PEGA", "AI", "IONQ",
    # ── Semiconductors ─────────────────────────────────────────────────────────
    "MCHP", "ADI", "NXPI", "ON", "SWKS", "QRVO", "MPWR", "SITM", "FORM",
    "COHU", "ACLS", "MKSI", "UCTT", "ONTO", "RTEC", "CEVA", "SLAB", "POWI",
    "DIOD", "VSH", "IPGP", "VIAV", "LITE", "RMBS", "LSCC", "ALGM", "WOLF",
    "ENTG", "AZTA", "CCMP",
    # ── Financials ─────────────────────────────────────────────────────────────
    "BRK-B", "V", "JPM", "MA", "BAC", "WFC", "GS", "MS", "SCHW", "BLK",
    "ICE", "CME", "SPGI", "MCO", "MSCI", "CB", "AXP", "PGR", "TRV", "ALL",
    "MET", "PRU", "AFL", "AIG", "HIG", "MMC", "AON", "WTW", "CINF", "RNR",
    "FDS", "SSNC", "FIS", "FISV", "GPN", "PYPL", "DFS", "SYF", "COF", "C",
    "USB", "RF", "CFG", "FITB", "HBAN", "KEY", "CMA", "ZION", "MTB", "SNV",
    "EWBC", "BOKF", "WTFC", "FHB", "HWC", "ONB", "NBTB", "GBCI", "WAFD",
    "COLB", "VOYA", "LNC", "GL", "BHF", "CNO", "AFG", "ERIE", "KMPR",
    "GSHD", "RYAN", "ACGL", "RE", "IBKR", "NDAQ", "CBOE", "LPLA", "RJF",
    "BK", "NTRS", "STT",
    # ── Healthcare / Large-cap Pharma ──────────────────────────────────────────
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN",
    "ISRG", "GILD", "VRTX", "REGN", "BSX", "EW", "SYK", "MDT", "ZTS", "IDXX",
    "DXCM", "PODD", "HOLX", "NTRA", "EHC", "CI", "CVS", "HUM", "ELV", "CNC",
    "MOH", "BMY", "MRNA", "BNTX", "BIIB", "ALNY", "SGEN", "MDGL", "SRPT",
    "JAZZ", "NBIX", "PRGO", "SUPN", "INCY", "EXAS", "UTHR", "ACAD", "IONS",
    "FOLD", "PTCT", "ARCT", "KYMR", "IMVT", "ARQT", "TGTX", "IMCR", "KPTI",
    "AKRO", "MIRM", "SAGE", "AXSM", "DVAX", "IOVA", "FATE", "EDIT", "BEAM",
    "NTLA", "CRSP", "PACB", "ILMN", "GKOS", "ALGN", "ATRC", "NVCR", "NVAX",
    "MEDP", "IQV", "HURN", "EXLS", "TDOC", "HIMS", "DOCS", "ABCL", "ADMA",
    # ── Consumer Discretionary ─────────────────────────────────────────────────
    "HD", "COST", "WMT", "MCD", "SBUX", "NKE", "TJX", "LOW", "BKNG", "ABNB",
    "MAR", "HLT", "MGM", "WYNN", "LVS", "DKNG", "RCL", "CCL", "NCLH", "DIS",
    "NFLX", "ROKU", "SPOT", "CMCSA", "CHTR", "T", "VZ", "TMUS", "YUM", "QSR",
    "DPZ", "CMG", "WING", "TXRH", "BLMN", "CAKE", "BJRI", "DRI", "EAT",
    "JACK", "WEN", "ORLY", "AZO", "AAP", "GPC", "LKQ", "CPRT", "CVNA", "AN",
    "PAG", "LAD", "ABG", "SAH", "PTON", "LULU", "ANF", "AEO", "GPS", "KSS",
    "M", "JWN", "DDS", "TGT", "DLTR", "DG", "FIVE", "BJ", "KR", "DKS",
    "EBAY", "W", "CHWY", "ETSY", "SHOP", "MELI", "UBER", "LYFT", "DASH",
    "RIVN", "LCID", "NIO", "LI", "XPEV", "GM", "F",
    # ── Consumer Staples ───────────────────────────────────────────────────────
    "PG", "KO", "PEP", "MO", "PM", "MDLZ", "GIS", "K", "CPB", "CAG",
    "SJM", "MKC", "HSY", "HRL", "TSN", "CL", "CHD", "CLX", "EL", "ULTA",
    "KDP", "STZ", "MNST", "SAM",
    # ── Industrials ────────────────────────────────────────────────────────────
    "CAT", "HON", "GE", "ETN", "EMR", "ITW", "NOC", "RTX", "GD", "LMT",
    "BA", "TDG", "TXT", "HII", "KTOS", "AXON", "PLTR", "SAIC", "LDOS", "BAH",
    "CACI", "UNP", "CSX", "NSC", "WAB", "TT", "JCI", "ROK", "PH", "DOV",
    "IR", "XYL", "WCN", "RSG", "WM", "CWST", "CLH", "CTAS", "VRSK", "ADP",
    "JBHT", "CHRW", "XPO", "UPS", "FDX", "EXPD", "SAIA", "ODFL", "WERN",
    "KNX", "AGCO", "DE", "CNH", "TTC", "URI", "PWR", "MTZ", "PRIM", "DY",
    "ABM", "CSWI", "KELYA", "ASGN", "KBR", "MYRG", "AEGN", "GLDD",
    # ── Energy ─────────────────────────────────────────────────────────────────
    "XOM", "CVX", "COP", "EOG", "SLB", "BKR", "HAL", "OXY", "DVN", "FANG",
    "PXD", "MRO", "APA", "HES", "MPC", "PSX", "VLO", "PBF", "WMB", "KMI",
    "OKE", "ET", "EPD", "MMP", "TRGP", "LNG", "CQP", "RUN", "ENPH", "FSLR",
    "ARRY", "SEDG",
    # ── Utilities ──────────────────────────────────────────────────────────────
    "NEE", "SO", "DUK", "D", "EXC", "AEP", "PCG", "EIX", "SRE", "WEC",
    "ES", "PPL", "CMS", "NI", "AES", "CEG", "PLUG", "BE", "BLDP", "NOVA",
    "MAXN", "SPWR", "NEP", "BEP", "AQN", "PNW", "IDA", "OTTR", "AVA",
    # ── Materials ──────────────────────────────────────────────────────────────
    "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "AA", "CLF", "NUE", "STLD",
    "CMC", "WOR", "RS", "VMC", "MLM", "FMC", "CF", "MOS", "NTR", "IP",
    "PKG", "SEE", "SLGN", "BERY", "SON", "GPK", "CCK", "BLL", "ATR",
    # ── Real Estate ────────────────────────────────────────────────────────────
    "PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "O", "VICI", "WELL", "DLR",
    "AVB", "EQR", "ESS", "UDR", "CPT", "MAA", "NNN", "ADC", "EPRT", "VTR",
    "HR", "DOC", "MPW", "PEAK", "OHI", "ARE", "BXP", "HIW", "PDM", "DEI",
    "CUZ", "SLG", "KRC", "EXR", "CUBE", "NSA", "GEO", "SAFE", "IIPR", "COLD",
    "NYMT", "NRZ", "RWT", "RITM", "MITT", "MFA", "TWO", "BXMT", "LADR",
    "KREF", "GPMT", "ARCC", "OBDC", "BXSL", "PFLT", "TRIN", "CSWC", "SLRC",
    # ── Defense / Aerospace ────────────────────────────────────────────────────
    "LMT", "RTX", "NOC", "GD", "BA", "LHX", "TDG", "BWXT", "DRS", "MOOG",
    "HEI", "SPR", "HXL", "HEICO", "AVAV",
    # ── Additional large-cap mixed ─────────────────────────────────────────────
    "RBLX", "U", "PLTK", "APPS", "GENI", "TTD", "PUBM", "DV", "IAS", "MGNI",
    "FICO", "VRNS", "BLKB", "PCOR", "GWRE", "RGEN", "NEOG", "AEIS", "UFPI",
    "BLDR", "MDC", "TMHC", "KBH", "BZH", "NVR", "PHM", "TOL", "DHI", "LEN",
    "MHO", "SNAP", "PINS", "MTCH", "IAC", "ZG", "OPEN", "RDFN",
]
