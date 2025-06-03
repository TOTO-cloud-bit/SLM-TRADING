"""
SLM TRADE - Module 6: Base de Données & Système Complet Final
===========================================================

Ce module final intègre tous les modules précédents avec:
- Base de données SQLite optimisée pour trading
- Cache intelligent Redis-like en mémoire
- Synchronisation multi-sources en temps réel
- API unifiée pour toutes les fonctionnalités
- Interface web complète
- Système de notifications avancé
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import yfinance as yf
from dataclasses import dataclass, asdict
import hashlib
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
from flask import Flask, jsonify, request, render_template_string
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradeSignal:
    """Structure pour les signaux de trading"""
    symbol: str
    timestamp: datetime
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    price: float
    confidence: float
    strategy: str
    timeframe: str
    indicators: Dict[str, float]
    risk_metrics: Dict[str, float]

@dataclass
class Portfolio:
    """Structure pour le portefeuille"""
    cash: float
    positions: Dict[str, Dict[str, float]]  # {symbol: {shares: x, avg_price: y}}
    total_value: float
    daily_pnl: float
    unrealized_pnl: float
    realized_pnl: float

class SLMCache:
    """Système de cache intelligent pour les données de trading"""
    
    def __init__(self, max_size: int = 10000, ttl: int = 300):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live en secondes
        self.access_count = {}
        self._lock = threading.RLock()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Génère une clé unique pour le cache"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache"""
        with self._lock:
            if key in self.cache:
                # Vérifier l'expiration
                if time.time() - self.timestamps[key] > self.ttl:
                    self._remove(key)
                    return None
                
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return self.cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Stocke une valeur dans le cache"""
        with self._lock:
            # Nettoyage si nécessaire
            if len(self.cache) >= self.max_size:
                self._cleanup()
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
            self.access_count[key] = 1
    
    def _remove(self, key: str) -> None:
        """Supprime une entrée du cache"""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
        self.access_count.pop(key, None)
    
    def _cleanup(self) -> None:
        """Nettoie le cache en supprimant les entrées les moins utilisées"""
        # Supprimer les entrées expirées
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        for key in expired_keys:
            self._remove(key)
        
        # Si encore trop d'entrées, supprimer les moins utilisées
        if len(self.cache) >= self.max_size:
            sorted_keys = sorted(self.access_count.items(), key=lambda x: x[1])
            keys_to_remove = [key for key, _ in sorted_keys[:len(sorted_keys)//4]]
            for key in keys_to_remove:
                self._remove(key)
    
    def clear(self) -> None:
        """Vide complètement le cache"""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()
            self.access_count.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache"""
        with self._lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': sum(self.access_count.values()) / max(len(self.cache), 1),
                'ttl': self.ttl
            }

class SLMDatabase:
    """Gestionnaire de base de données optimisé pour le trading"""
    
    def __init__(self, db_path: str = "slm_trade.db"):
        self.db_path = db_path
        self.cache = SLMCache()
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialise la structure de la base de données"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Table pour les données OHLCV
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    timeframe TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    adj_close REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, timeframe)
                )
            """)
            
            # Table pour les signaux de trading
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    signal_type TEXT NOT NULL,
                    price REAL NOT NULL,
                    confidence REAL NOT NULL,
                    strategy TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    indicators TEXT,  -- JSON
                    risk_metrics TEXT,  -- JSON
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table pour les trades exécutés
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS executed_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    trade_type TEXT NOT NULL,  -- 'BUY' or 'SELL'
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    commission REAL DEFAULT 0,
                    strategy TEXT,
                    pnl REAL DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table pour le portefeuille
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    cash REAL NOT NULL,
                    total_value REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    positions TEXT,  -- JSON
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table pour les métriques de performance
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    symbol TEXT,
                    strategy TEXT,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    var_95 REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, symbol, strategy)
                )
            """)
            
            # Index pour optimiser les requêtes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON trading_signals(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON executed_trades(symbol, timestamp)")
            
            conn.commit()
            logger.info("Base de données initialisée avec succès")
    
    def store_market_data(self, symbol: str, data: pd.DataFrame, timeframe: str) -> None:
        """Stocke les données de marché"""
        with sqlite3.connect(self.db_path) as conn:
            try:
                records = []
                for timestamp, row in data.iterrows():
                    records.append((
                        symbol, timestamp, timeframe,
                        float(row['Open']), float(row['High']), 
                        float(row['Low']), float(row['Close']),
                        int(row['Volume']), 
                        float(row.get('Adj Close', row['Close']))
                    ))
                
                conn.executemany("""
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timestamp, timeframe, open, high, low, close, volume, adj_close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                
                conn.commit()
                logger.info(f"Stocké {len(records)} enregistrements pour {symbol} ({timeframe})")
                
            except Exception as e:
                logger.error(f"Erreur lors du stockage des données de marché: {e}")
    
    def get_market_data(self, symbol: str, timeframe: str, 
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Récupère les données de marché avec cache"""
        cache_key = self.cache._generate_key(symbol, timeframe, start_date, end_date)
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT timestamp, open, high, low, close, volume, adj_close
                FROM market_data
                WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, timeframe]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp"
            
            df = pd.read_sql_query(query, conn, params=params, 
                                 parse_dates=['timestamp'], index_col='timestamp')
            
            if not df.empty:
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
                self.cache.set(cache_key, df)
            
            return df
    
    def store_signal(self, signal: TradeSignal) -> None:
        """Stocke un signal de trading"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trading_signals 
                (symbol, timestamp, signal_type, price, confidence, strategy, timeframe, indicators, risk_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.symbol, signal.timestamp, signal.signal_type,
                signal.price, signal.confidence, signal.strategy, signal.timeframe,
                json.dumps(signal.indicators), json.dumps(signal.risk_metrics)
            ))
            conn.commit()
    
    def store_trade(self, symbol: str, trade_type: str, quantity: float, 
                   price: float, timestamp: datetime, commission: float = 0,
                   strategy: str = None, pnl: float = 0) -> None:
        """Stocke un trade exécuté"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO executed_trades 
                (symbol, trade_type, quantity, price, timestamp, commission, strategy, pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, trade_type, quantity, price, timestamp, commission, strategy, pnl))
            conn.commit()
    
    def store_portfolio_snapshot(self, portfolio: Portfolio, timestamp: datetime) -> None:
        """Stocke un snapshot du portefeuille"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO portfolio_history 
                (timestamp, cash, total_value, daily_pnl, unrealized_pnl, realized_pnl, positions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, portfolio.cash, portfolio.total_value,
                portfolio.daily_pnl, portfolio.unrealized_pnl, 
                portfolio.realized_pnl, json.dumps(portfolio.positions)
            ))
            conn.commit()
    
    def get_latest_signals(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Récupère les derniers signaux"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM trading_signals
                WHERE 1=1
            """
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            columns = [description[0] for description in cursor.description]
            
            signals = []
            for row in cursor.fetchall():
                signal_dict = dict(zip(columns, row))
                # Parse JSON fields
                signal_dict['indicators'] = json.loads(signal_dict.get('indicators', '{}'))
                signal_dict['risk_metrics'] = json.loads(signal_dict.get('risk_metrics', '{}'))
                signals.append(signal_dict)
            
            return signals
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Génère un résumé de performance"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Total des trades
            total_trades = conn.execute("""
                SELECT COUNT(*) FROM executed_trades 
                WHERE timestamp >= ?
            """, (start_date,)).fetchone()[0]
            
            # PnL total
            total_pnl = conn.execute("""
                SELECT COALESCE(SUM(pnl), 0) FROM executed_trades 
                WHERE timestamp >= ?
            """, (start_date,)).fetchone()[0]
            
            # Win rate
            winning_trades = conn.execute("""
                SELECT COUNT(*) FROM executed_trades 
                WHERE timestamp >= ? AND pnl > 0
            """, (start_date,)).fetchone()[0]
            
            win_rate = (winning_trades / max(total_trades, 1)) * 100
            
            # Portfolio evolution
            portfolio_data = pd.read_sql_query("""
                SELECT timestamp, total_value FROM portfolio_history 
                WHERE timestamp >= ? ORDER BY timestamp
            """, conn, params=(start_date,), parse_dates=['timestamp'])
            
            max_value = portfolio_data['total_value'].max() if not portfolio_data.empty else 0
            min_value = portfolio_data['total_value'].min() if not portfolio_data.empty else 0
            current_value = portfolio_data['total_value'].iloc[-1] if not portfolio_data.empty else 0
            
            return {
                'period_days': days,
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'current_portfolio_value': current_value,
                'max_portfolio_value': max_value,
                'min_portfolio_value': min_value,
                'drawdown_pct': ((max_value - current_value) / max(max_value, 1)) * 100
            }

class SLMDataSyncManager:
    """Gestionnaire de synchronisation multi-sources"""
    
    def __init__(self, database: SLMDatabase):
        self.database = database
        self.active_symbols = set()
        self.sync_threads = {}
        self.running = False
        self.update_callbacks = []
    
    def add_symbol(self, symbol: str, timeframes: List[str] = ['1h', '1d']) -> None:
        """Ajoute un symbole à la synchronisation"""
        self.active_symbols.add(symbol)
        
        if self.running:
            self._start_sync_thread(symbol, timeframes)
        
        logger.info(f"Symbole {symbol} ajouté à la synchronisation")
    
    def remove_symbol(self, symbol: str) -> None:
        """Retire un symbole de la synchronisation"""
        if symbol in self.active_symbols:
            self.active_symbols.remove(symbol)
            
            if symbol in self.sync_threads:
                # Arrêter le thread de synchronisation
                self.sync_threads[symbol]['stop_event'].set()
                del self.sync_threads[symbol]
        
        logger.info(f"Symbole {symbol} retiré de la synchronisation")
    
    def start_sync(self) -> None:
        """Démarre la synchronisation en temps réel"""
        self.running = True
        
        for symbol in self.active_symbols:
            self._start_sync_thread(symbol, ['1h', '1d'])
        
        logger.info("Synchronisation en temps réel démarrée")
    
    def stop_sync(self) -> None:
        """Arrête la synchronisation"""
        self.running = False
        
        for symbol, thread_info in self.sync_threads.items():
            thread_info['stop_event'].set()
        
        self.sync_threads.clear()
        logger.info("Synchronisation arrêtée")
    
    def _start_sync_thread(self, symbol: str, timeframes: List[str]) -> None:
        """Démarre un thread de synchronisation pour un symbole"""
        if symbol in self.sync_threads:
            return
        
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._sync_worker,
            args=(symbol, timeframes, stop_event),
            daemon=True
        )
        
        self.sync_threads[symbol] = {
            'thread': thread,
            'stop_event': stop_event
        }
        
        thread.start()
    
    def _sync_worker(self, symbol: str, timeframes: List[str], stop_event: threading.Event) -> None:
        """Worker de synchronisation pour un symbole"""
        while not stop_event.is_set():
            try:
                ticker = yf.Ticker(symbol)
                
                for timeframe in timeframes:
                    # Récupérer les données récentes
                    data = ticker.history(period="5d", interval=timeframe)
                    
                    if not data.empty:
                        # Stocker en base
                        self.database.store_market_data(symbol, data, timeframe)
                        
                        # Notifier les callbacks
                        for callback in self.update_callbacks:
                            try:
                                callback(symbol, timeframe, data)
                            except Exception as e:
                                logger.error(f"Erreur dans callback: {e}")
                
                # Attendre avant la prochaine mise à jour
                stop_event.wait(60)  # 1 minute
                
            except Exception as e:
                logger.error(f"Erreur de synchronisation pour {symbol}: {e}")
                stop_event.wait(30)  # Attendre 30s avant de retry
    
    def add_update_callback(self, callback) -> None:
        """Ajoute un callback pour les mises à jour de données"""
        self.update_callbacks.append(callback)

class SLMTradeSystemComplete:
    """Système de trading complet SLM TRADE - Tous modules intégrés"""
    
    def __init__(self, initial_capital: float = 100000):
        # Initialisation des composants
        self.database = SLMDatabase()
        self.sync_manager = SLMDataSyncManager(self.database)
        
        # Portfolio
        self.portfolio = Portfolio(
            cash=initial_capital,
            positions={},
            total_value=initial_capital,
            daily_pnl=0,
            unrealized_pnl=0,
            realized_pnl=0
        )
        
        # Composants des modules précédents
        from datetime import datetime
        import yfinance as yf
        
        # Configuration
        self.commission_rate = 0.001  # 0.1%
        self.max_risk_per_trade = 0.02  # 2%
        self.active_strategies = ['rsi_mean_reversion', 'ma_crossover', 'bollinger_bounce']
        
        # Threads et état
        self.trading_active = False
        self.monitoring_thread = None
        
        logger.info("Système SLM TRADE initialisé avec succès")
    
    def add_symbol_to_watch(self, symbol: str) -> None:
        """Ajoute un symbole à la surveillance"""
        self.sync_manager.add_symbol(symbol)
        logger.info(f"Symbole {symbol} ajouté à la surveillance")
    
    def start_live_trading(self) -> None:
        """Démarre le trading en temps réel"""
        if self.trading_active:
            logger.warning("Le trading est déjà actif")
            return
        
        self.trading_active = True
        
        # Démarrer la synchronisation des données
        self.sync_manager.start_sync()
        
        # Démarrer le monitoring
        self.monitoring_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Trading en temps réel démarré")
    
    def stop_live_trading(self) -> None:
        """Arrête le trading en temps réel"""
        self.trading_active = False
        self.sync_manager.stop_sync()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Trading en temps réel arrêté")
    
    def _trading_loop(self) -> None:
        """Boucle principale de trading"""
        while self.trading_active:
            try:
                # Analyser chaque symbole surveillé
                for symbol in self.sync_manager.active_symbols:
                    signals = self._generate_signals(symbol)
                    
                    for signal in signals:
                        if signal.confidence > 0.7:  # Seuil de confiance
                            self._process_signal(signal)
                
                # Mise à jour du portefeuille
                self._update_portfolio()
                
                # Sauvegarde périodique
                self.database.store_portfolio_snapshot(self.portfolio, datetime.now())
                
                time.sleep(30)  # Attendre 30 secondes
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle de trading: {e}")
                time.sleep(60)
    
    def _generate_signals(self, symbol: str) -> List[TradeSignal]:
        """Génère des signaux de trading pour un symbole"""
        signals = []
        
        try:
            # Récupérer les données
            data_1h = self.database.get_market_data(symbol, '1h')
            data_1d = self.database.get_market_data(symbol, '1d')
            
            if data_1h.empty or data_1d.empty:
                return signals
            
            current_price = data_1h['Close'].iloc[-1]
            
            # Calculer les indicateurs
            indicators = self._calculate_indicators(data_1h)
            risk_metrics = self._calculate_risk_metrics(symbol, data_1d)
            
            # Stratégies de trading
            for strategy in self.active_strategies:
                signal_type, confidence = self._evaluate_strategy(strategy, indicators, data_1h)
                
                if signal_type != 'HOLD':
                    signal = TradeSignal(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        signal_type=signal_type,
                        price=current_price,
                        confidence=confidence,
                        strategy=strategy,
                        timeframe='1h',
                        indicators=indicators,
                        risk_metrics=risk_metrics
                    )
                    signals.append(signal)
                    
                    # Stocker le signal
                    self.database.store_signal(signal)
        
        except Exception as e:
            logger.error(f"Erreur génération signaux pour {symbol}: {e}")
        
        return signals
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calcule les indicateurs techniques"""
        if len(data) < 50:
            return {}
        
        indicators = {}
        
        try:
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # Moyennes mobiles
            indicators['sma_20'] = data['Close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = data['Close'].rolling(50).mean().iloc[-1]
            indicators['ema_12'] = data['Close'].ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = data['Close'].ewm(span=26).mean().iloc[-1]
            
            # MACD
            macd_line = indicators['ema_12'] - indicators['ema_26']
            signal_line = data['Close'].ewm(span=9).mean().iloc[-1]
            indicators['macd'] = macd_line
            indicators['macd_signal'] = signal_line
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            sma = data['Close'].rolling(bb_period).mean()
            std = data['Close'].rolling(bb_period).std()
            indicators['bb_upper'] = (sma + (std * bb_std)).iloc[-1]
            indicators['bb_lower'] = (sma - (std * bb_std)).iloc[-1]
            indicators['bb_middle'] = sma.iloc[-1]
            
            # Prix actuel
            indicators['current_price'] = data['Close'].iloc[-1]
            
        except Exception as e:
            logger.error(f"Erreur calcul indicateurs: {e}")
        
        return indicators
    
    def _calculate_risk_metrics(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        """Calcule les métriques de risque"""
        if len(data) < 30:
            return {}
        
        metrics = {}
        
        try:
            returns = data['Close'].pct_change().dropna()
            
            # Volatilité
            metrics['volatility'] = returns.std() * np.sqrt(252)
            
            # VaR 95%
            metrics['var_95'] = np.percentile(returns, 5)
            
            # Maximum Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()
            
            # Beta (vs SPY approximation)
            metrics['beta'] = 1.0  # Approximation
            
        except Exception as e:
            logger.error(f"Erreur calcul métriques risque: {e}")
        
        return metrics
    
    def _evaluate_strategy(self, strategy: str, indicators: Dict[str, float], data: pd.DataFrame) -> Tuple[str, float]:
        """Évalue une stratégie de trading"""
        if not indicators:
            return 'HOLD', 0.0
        
        try:
            if strategy == 'rsi_mean_reversion':
                rsi = indicators.get('rsi', 50)
                if rsi < 30:
                    return 'BUY', min(0.8, (30 - rsi) / 20)
                elif rsi > 70:
                    return 'SELL', min(0.8, (rsi - 70) / 20)
            
            elif strategy == 'ma_crossover':
                sma_20 = indicators.get('sma_20', 0)
                sma_50 = indicators.get('sma_50', 0)
                current_price = indicators.get('current_price', 0)
                
                if sma_20 > sma_50 and current_price > sma_20:
                    return 'BUY', 0.6
                elif sma_20 < sma_50 and current_price < sma_20:
                    return 'SELL', 0.6
            
            elif strategy == 'bollinger_bounce':
                current_price = indicators.get('current_price', 0)
                bb_upper = indicators.get('bb_upper', 0)
                bb_lower = indicators.get('bb_lower', 0)
                bb_middle = indicators.get('bb_middle', 0)
                
                if current_price <= bb_lower:
                    return 'BUY', 0.7
                elif current_price >= bb_upper:
                    return 'SELL', 0.7
        
        except Exception as e:
            logger.error(f"Erreur évaluation stratégie {strategy}: {e}")
        
        return 'HOLD', 0.0
    
    def _process_signal(self, signal: TradeSignal) -> None:
        """Traite un signal de trading et exécute si nécessaire"""
        try:
            # Vérifier si on a déjà une position
            current_position = self.portfolio.positions.get(signal.symbol, {'shares': 0, 'avg_price': 0})
            
            # Calculer la taille de position
            risk_amount = self.portfolio.total_value * self.max_risk_per_trade
            volatility = signal.risk_metrics.get('volatility', 0.2)
            
            # Position sizing basé sur la volatilité
            position_size = min(
                int(risk_amount / (signal.price * volatility)),
                int(self.portfolio.cash / signal.price * 0.25)  # Max 25% du cash
            )
            
            if signal.signal_type == 'BUY' and current_position['shares'] <= 0 and position_size > 0:
                self._execute_buy(signal.symbol, position_size, signal.price, signal.strategy)
                
            elif signal.signal_type == 'SELL' and current_position['shares'] > 0:
                self._execute_sell(signal.symbol, current_position['shares'], signal.price, signal.strategy)
                
        except Exception as e:
            logger.error(f"Erreur traitement signal: {e}")
    
    def _execute_buy(self, symbol: str, quantity: int, price: float, strategy: str) -> None:
        """Exécute un ordre d'achat"""
        try:
            total_cost = quantity * price
            commission = total_cost * self.commission_rate
            total_with_commission = total_cost + commission
            
            if self.portfolio.cash >= total_with_commission:
                # Mise à jour du portefeuille
                self.portfolio.cash -= total_with_commission
                
                if symbol in self.portfolio.positions:
                    # Position existante - moyenne pondérée
                    old_shares = self.portfolio.positions[symbol]['shares']
                    old_price = self.portfolio.positions[symbol]['avg_price']
                    
                    new_shares = old_shares + quantity
                    new_avg_price = ((old_shares * old_price) + total_cost) / new_shares
                    
                    self.portfolio.positions[symbol] = {
                        'shares': new_shares,
                        'avg_price': new_avg_price
                    }
                else:
                    # Nouvelle position
                    self.portfolio.positions[symbol] = {
                        'shares': quantity,
                        'avg_price': price
                    }
                
                # Enregistrer le trade
                self.database.store_trade(
                    symbol=symbol,
                    trade_type='BUY',
                    quantity=quantity,
                    price=price,
                    timestamp=datetime.now(),
                    commission=commission,
                    strategy=strategy
                )
                
                logger.info(f"ACHAT exécuté: {quantity} {symbol} @ {price:.2f}")
            
        except Exception as e:
            logger.error(f"Erreur exécution achat: {e}")
    
    def _execute_sell(self, symbol: str, quantity: int, price: float, strategy: str) -> None:
        """Exécute un ordre de vente"""
        try:
            if symbol in self.portfolio.positions and self.portfolio.positions[symbol]['shares'] >= quantity:
                total_proceeds = quantity * price
                commission = total_proceeds * self.commission_rate
                net_proceeds = total_proceeds - commission
                
                # Calculer le PnL
                avg_price = self.portfolio.positions[symbol]['avg_price']
                pnl = (price - avg_price) * quantity - commission
                
                # Mise à jour du portefeuille
                self.portfolio.cash += net_proceeds
                self.portfolio.realized_pnl += pnl
                
                # Mise à jour de la position
                remaining_shares = self.portfolio.positions[symbol]['shares'] - quantity
                if remaining_shares > 0:
                    self.portfolio.positions[symbol]['shares'] = remaining_shares
                else:
                    del self.portfolio.positions[symbol]
                
                # Enregistrer le trade
                self.database.store_trade(
                    symbol=symbol,
                    trade_type='SELL',
                    quantity=quantity,
                    price=price,
                    timestamp=datetime.now(),
                    commission=commission,
                    strategy=strategy,
                    pnl=pnl
                )
                
                logger.info(f"VENTE exécutée: {quantity} {symbol} @ {price:.2f} | PnL: {pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Erreur exécution vente: {e}")
    
    def _update_portfolio(self) -> None:
        """Met à jour la valeur du portefeuille"""
        try:
            total_value = self.portfolio.cash
            unrealized_pnl = 0
            
            for symbol, position in self.portfolio.positions.items():
                try:
                    # Récupérer le prix actuel
                    recent_data = self.database.get_market_data(symbol, '1h')
                    if not recent_data.empty:
                        current_price = recent_data['Close'].iloc[-1]
                        position_value = position['shares'] * current_price
                        total_value += position_value
                        
                        # PnL non réalisé
                        position_pnl = (current_price - position['avg_price']) * position['shares']
                        unrealized_pnl += position_pnl
                        
                except Exception as e:
                    logger.warning(f"Erreur mise à jour prix {symbol}: {e}")
            
            # Calculer le PnL journalier
            previous_value = self.portfolio.total_value
            daily_pnl = total_value - previous_value
            
            # Mise à jour
            self.portfolio.total_value = total_value
            self.portfolio.unrealized_pnl = unrealized_pnl
            self.portfolio.daily_pnl = daily_pnl
            
        except Exception as e:
            logger.error(f"Erreur mise à jour portefeuille: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Génère les données pour le dashboard"""
        try:
            # Performance summary
            performance = self.database.get_performance_summary(30)
            
            # Signaux récents
            recent_signals = self.database.get_latest_signals(limit=20)
            
            # Portfolio current state
            portfolio_data = {
                'cash': self.portfolio.cash,
                'total_value': self.portfolio.total_value,
                'daily_pnl': self.portfolio.daily_pnl,
                'unrealized_pnl': self.portfolio.unrealized_pnl,
                'realized_pnl': self.portfolio.realized_pnl,
                'positions': self.portfolio.positions,
                'position_count': len(self.portfolio.positions)
            }
            
            # Cache stats
            cache_stats = self.database.cache.stats()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'performance': performance,
                'portfolio': portfolio_data,
                'recent_signals': recent_signals,
                'cache_stats': cache_stats,
                'active_symbols': list(self.sync_manager.active_symbols),
                'trading_active': self.trading_active
            }
            
        except Exception as e:
            logger.error(f"Erreur génération dashboard: {e}")
            return {}
    
    def create_performance_chart(self, days: int = 30) -> str:
        """Crée un graphique de performance"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            with sqlite3.connect(self.database.db_path) as conn:
                portfolio_data = pd.read_sql_query("""
                    SELECT timestamp, total_value, daily_pnl, unrealized_pnl
                    FROM portfolio_history 
                    WHERE timestamp >= ? 
                    ORDER BY timestamp
                """, conn, params=(start_date,), parse_dates=['timestamp'])
            
            if portfolio_data.empty:
                return "<div>Pas de données de performance disponibles</div>"
            
            # Créer le graphique
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Valeur du Portefeuille', 'PnL Journalier'],
                vertical_spacing=0.1
            )
            
            # Courbe de valeur du portefeuille
            fig.add_trace(
                go.Scatter(
                    x=portfolio_data['timestamp'],
                    y=portfolio_data['total_value'],
                    mode='lines',
                    name='Valeur Totale',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # PnL journalier
            colors = ['green' if x >= 0 else 'red' for x in portfolio_data['daily_pnl']]
            fig.add_trace(
                go.Bar(
                    x=portfolio_data['timestamp'],
                    y=portfolio_data['daily_pnl'],
                    name='PnL Journalier',
                    marker_color=colors
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title='Performance du Portefeuille SLM TRADE',
                height=600,
                showlegend=True
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Erreur création graphique: {e}")
            return f"<div>Erreur génération graphique: {e}</div>"

# Interface Web Flask
class SLMWebInterface:
    """Interface web pour SLM TRADE"""
    
    def __init__(self, trading_system: SLMTradeSystemComplete):
        self.trading_system = trading_system
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Configure les routes de l'interface web"""
        
        @self.app.route('/')
        def dashboard():
            """Page principale du dashboard"""
            dashboard_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>SLM TRADE - Dashboard</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
                    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
                    .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .card h3 { margin-top: 0; color: #34495e; }
                    .value { font-size: 24px; font-weight: bold; }
                    .positive { color: #27ae60; }
                    .negative { color: #e74c3c; }
                    .neutral { color: #95a5a6; }
                    .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
                    .signals-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
                    .signals-table th, .signals-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                    .signals-table th { background-color: #f8f9fa; }
                    .signal-buy { color: #27ae60; font-weight: bold; }
                    .signal-sell { color: #e74c3c; font-weight: bold; }
                    .signal-hold { color: #95a5a6; }
                    .controls { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
                    .btn { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; margin-right: 10px; }
                    .btn-primary { background: #3498db; color: white; }
                    .btn-success { background: #27ae60; color: white; }
                    .btn-danger { background: #e74c3c; color: white; }
                    .btn:hover { opacity: 0.8; }
                    input[type="text"] { padding: 8px; border: 1px solid #ddd; border-radius: 4px; margin-right: 10px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚀 SLM TRADE - Système de Trading Algorithmique</h1>
                        <p>Dashboard de Trading Professionnel avec IA</p>
                    </div>
                    
                    <div class="controls">
                        <h3>Contrôles du Système</h3>
                        <button class="btn btn-success" onclick="startTrading()">▶️ Démarrer Trading</button>
                        <button class="btn btn-danger" onclick="stopTrading()">⏹️ Arrêter Trading</button>
                        <input type="text" id="symbolInput" placeholder="Symbole (ex: AAPL)">
                        <button class="btn btn-primary" onclick="addSymbol()">➕ Ajouter Symbole</button>
                        <button class="btn btn-primary" onclick="refreshData()">🔄 Actualiser</button>
                    </div>
                    
                    <div class="stats-grid" id="statsGrid">
                        <!-- Stats will be loaded here -->
                    </div>
                    
                    <div class="chart-container">
                        <h3>📈 Performance du Portefeuille</h3>
                        <div id="performanceChart"></div>
                    </div>
                    
                    <div class="card">
                        <h3>🎯 Signaux de Trading Récents</h3>
                        <div id="signalsTable"></div>
                    </div>
                </div>
                
                <script>
                    function loadDashboard() {
                        fetch('/api/dashboard')
                            .then(response => response.json())
                            .then(data => updateDashboard(data))
                            .catch(error => console.error('Error:', error));
                    }
                    
                    function updateDashboard(data) {
                        // Update stats
                        const statsGrid = document.getElementById('statsGrid');
                        const portfolio = data.portfolio;
                        const performance = data.performance;
                        
                        statsGrid.innerHTML = `
                            <div class="card">
                                <h3>💰 Valeur Totale</h3>
                                <div class="value">${portfolio.total_value.toLocaleString('fr-FR', {style: 'currency', currency: 'USD'})}</div>
                            </div>
                            <div class="card">
                                <h3>📊 PnL Journalier</h3>
                                <div class="value ${portfolio.daily_pnl >= 0 ? 'positive' : 'negative'}">
                                    ${portfolio.daily_pnl.toLocaleString('fr-FR', {style: 'currency', currency: 'USD'})}
                                </div>
                            </div>
                            <div class="card">
                                <h3>💸 Liquidités</h3>
                                <div class="value">${portfolio.cash.toLocaleString('fr-FR', {style: 'currency', currency: 'USD'})}</div>
                            </div>
                            <div class="card">
                                <h3>📈 PnL Non Réalisé</h3>
                                <div class="value ${portfolio.unrealized_pnl >= 0 ? 'positive' : 'negative'}">
                                    ${portfolio.unrealized_pnl.toLocaleString('fr-FR', {style: 'currency', currency: 'USD'})}
                                </div>
                            </div>
                            <div class="card">
                                <h3>🎯 Win Rate</h3>
                                <div class="value">${performance.win_rate.toFixed(1)}%</div>
                            </div>
                            <div class="card">
                                <h3>📊 Positions Actives</h3>
                                <div class="value">${portfolio.position_count}</div>
                            </div>
                        `;
                        
                        // Update signals table
                        const signalsTable = document.getElementById('signalsTable');
                        let tableHTML = '<table class="signals-table"><tr><th>Symbole</th><th>Signal</th><th>Prix</th><th>Confiance</th><th>Stratégie</th><th>Timestamp</th></tr>';
                        
                        data.recent_signals.slice(0, 10).forEach(signal => {
                            const signalClass = signal.signal_type === 'BUY' ? 'signal-buy' : 
                                              signal.signal_type === 'SELL' ? 'signal-sell' : 'signal-hold';
                            tableHTML += `
                                <tr>
                                    <td>${signal.symbol}</td>
                                    <td class="${signalClass}">${signal.signal_type}</td>
                                    <td>${signal.price.toFixed(2)}</td>
                                    <td>${(signal.confidence * 100).toFixed(1)}%</td>
                                    <td>${signal.strategy}</td>
                                    <td>${new Date(signal.timestamp).toLocaleString()}</td>
                                </tr>
                            `;
                        });
                        tableHTML += '</table>';
                        signalsTable.innerHTML = tableHTML;
                    }
                    
                    function startTrading() {
                        fetch('/api/start-trading', {method: 'POST'})
                            .then(response => response.json())
                            .then(data => {
                                alert(data.message);
                                loadDashboard();
                            });
                    }
                    
                    function stopTrading() {
                        fetch('/api/stop-trading', {method: 'POST'})
                            .then(response => response.json())
                            .then(data => {
                                alert(data.message);
                                loadDashboard();
                            });
                    }
                    
                    function addSymbol() {
                        const symbol = document.getElementById('symbolInput').value.toUpperCase();
                        if (symbol) {
                            fetch('/api/add-symbol', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({symbol: symbol})
                            })
                            .then(response => response.json())
                            .then(data => {
                                alert(data.message);
                                document.getElementById('symbolInput').value = '';
                                loadDashboard();
                            });
                        }
                    }
                    
                    function refreshData() {
                        loadDashboard();
                    }
                    
                    function loadPerformanceChart() {
                        fetch('/api/performance-chart')
                            .then(response => response.text())
                            .then(html => {
                                document.getElementById('performanceChart').innerHTML = html;
                            });
                    }
                    
                    // Auto-refresh every 30 seconds
                    setInterval(loadDashboard, 30000);
                    
                    // Initial load
                    loadDashboard();
                    loadPerformanceChart();
                </script>
            </body>
            </html>
            """
            return dashboard_html
        
        @self.app.route('/api/dashboard')
        def api_dashboard():
            """API endpoint pour les données du dashboard"""
            return jsonify(self.trading_system.get_dashboard_data())
        
        @self.app.route('/api/start-trading', methods=['POST'])
        def api_start_trading():
            """API endpoint pour démarrer le trading"""
            try:
                self.trading_system.start_live_trading()
                return jsonify({'status': 'success', 'message': 'Trading démarré avec succès'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})
        
        @self.app.route('/api/stop-trading', methods=['POST'])
        def api_stop_trading():
            """API endpoint pour arrêter le trading"""
            try:
                self.trading_system.stop_live_trading()
                return jsonify({'status': 'success', 'message': 'Trading arrêté avec succès'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})
        
        @self.app.route('/api/add-symbol', methods=['POST'])
        def api_add_symbol():
            """API endpoint pour ajouter un symbole"""
            try:
                data = request.get_json()
                symbol = data.get('symbol', '').upper()
                if symbol:
                    self.trading_system.add_symbol_to_watch(symbol)
                    return jsonify({'status': 'success', 'message': f'Symbole {symbol} ajouté'})
                else:
                    return jsonify({'status': 'error', 'message': 'Symbole invalide'})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})
        
        @self.app.route('/api/performance-chart')
        def api_performance_chart():
            """API endpoint pour le graphique de performance"""
            return self.trading_system.create_performance_chart()
    
    def run(self, host='localhost', port=5000, debug=False):
        """Lance l'interface web"""
        logger.info(f"Interface web démarrée sur http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Fonction principale d'initialisation
def initialize_slm_trade_complete(initial_capital: float = 100000, 
                                web_interface: bool = True) -> SLMTradeSystemComplete:
    """
    Initialise le système complet SLM TRADE
    
    Args:
        initial_capital: Capital initial pour le trading
        web_interface: Si True, lance l'interface web
    
    Returns:
        Instance du système de trading complet
    """
    print("🚀 Initialisation du Système SLM TRADE Complet...")
    print("=" * 60)
    
    # Créer le système de trading
    trading_system = SLMTradeSystemComplete(initial_capital)
    
    # Ajouter quelques symboles populaires par défaut
    default_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    for symbol in default_symbols:
        trading_system.add_symbol_to_watch(symbol)
    
    print(f"✅ Système initialisé avec {len(default_symbols)} symboles")
    print(f"💰 Capital initial: ${initial_capital:,.2f}")
    print(f"🎯 Stratégies actives: {len(trading_system.active_strategies)}")
    
    # Lancer l'interface web si demandée
    if web_interface:
        web_app = SLMWebInterface(trading_system)
        print("\n🌐 Interface Web disponible sur: http://localhost:5000")
        print("📊 Dashboard en temps réel avec toutes les métriques")
        
        # Lancer dans un thread séparé pour ne pas bloquer
        web_thread = threading.Thread(
            target=lambda: web_app.run(host='0.0.0.0', port=5000, debug=False),
            daemon=True
        )
        web_thread.start()
        
        print("\n⚡ Pour démarrer le trading automatique, utilisez l'interface web ou:")
        print("   trading_system.start_live_trading()")
    
    print("\n🎉 Système SLM TRADE prêt à l'utilisation!")
    print("=" * 60)
    
    return trading_system

# Exemple d'utilisation complète
if __name__ == "__main__":
    # Initialiser le système complet
    slm_system = initialize_slm_trade_complete(
        initial_capital=100000,
        web_interface=True
    )
    
    # Garder le programme en vie
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt du système...")
        slm_system.stop_live_trading()
        print("✅ Système arrêté proprement")