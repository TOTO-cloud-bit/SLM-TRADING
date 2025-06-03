"""
SLM TRADE - Module 4: Stratégies de Trading Algorithmiques & IA
================================================================

Ce module implémente des stratégies de trading avancées avec:
- Machine Learning pour la prédiction de prix
- Backtesting sophistiqué avec métriques détaillées
- Optimisation automatique des paramètres
- Signaux multi-timeframes
- Exécution automatique des trades
- Stratégies adaptatives basées sur l'IA

Auteur: Assistant IA
Version: 4.0
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SLMAlgorithmicTrader:
    """
    Système de trading algorithmique avancé avec IA
    """
    
    def __init__(self):
        self.strategies = {}
        self.models = {}
        self.scalers = {}
        self.backtest_results = {}
        self.active_positions = {}
        self.performance_metrics = {}
        
    def fetch_data(self, symbol, period="2y", interval="1d"):
        """Récupère les données de marché"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"❌ Aucune donnée trouvée pour {symbol}")
                return None
                
            # Calcul des indicateurs techniques
            data = self.calculate_technical_indicators(data)
            print(f"✅ Données récupérées pour {symbol}: {len(data)} barres")
            return data
            
        except Exception as e:
            print(f"❌ Erreur lors de la récupération des données: {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calcule tous les indicateurs techniques nécessaires"""
        # Moyennes mobiles
        data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
        data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
        data['EMA_12'] = ta.trend.ema_indicator(data['Close'], window=12)
        data['EMA_26'] = ta.trend.ema_indicator(data['Close'], window=26)
        
        # MACD
        data['MACD'] = ta.trend.macd_diff(data['Close'])
        data['MACD_signal'] = ta.trend.macd_signal(data['Close'])
        
        # RSI
        data['RSI'] = ta.momentum.rsi(data['Close'])
        
        # Bollinger Bands
        data['BB_upper'] = ta.volatility.bollinger_hband(data['Close'])
        data['BB_lower'] = ta.volatility.bollinger_lband(data['Close'])
        data['BB_middle'] = ta.volatility.bollinger_mavg(data['Close'])
        
        # Stochastic
        data['Stoch_K'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
        data['Stoch_D'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'])
        
        # ATR
        data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        
        # Momentum
        data['ROC'] = ta.momentum.roc(data['Close'], window=12)
        data['Williams_R'] = ta.momentum.williams_r(data['High'], data['Low'], data['Close'])
        
        # Volatilité
        data['Close_returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Close_returns'].rolling(window=20).std()
        
        return data
    
    def prepare_ml_features(self, data):
        """Prépare les features pour le machine learning"""
        features = []
        
        # Features techniques
        feature_columns = [
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal',
            'RSI', 'BB_upper', 'BB_lower', 'Stoch_K', 'Stoch_D', 'ATR',
            'Volume_SMA', 'OBV', 'ROC', 'Williams_R', 'Volatility'
        ]
        
        # Ratios et relations
        data['Price_to_SMA20'] = data['Close'] / data['SMA_20']
        data['Price_to_SMA50'] = data['Close'] / data['SMA_50']
        data['BB_position'] = (data['Close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])
        data['Volume_ratio'] = data['Volume'] / data['Volume_SMA']
        
        feature_columns.extend(['Price_to_SMA20', 'Price_to_SMA50', 'BB_position', 'Volume_ratio'])
        
        # Features de lag
        for col in ['Close', 'Volume', 'RSI', 'MACD']:
            for lag in [1, 2, 3, 5]:
                data[f'{col}_lag_{lag}'] = data[col].shift(lag)
                feature_columns.append(f'{col}_lag_{lag}')
        
        # Target: rendement futur
        data['Future_return'] = data['Close'].shift(-1) / data['Close'] - 1
        
        # Nettoyage des NaN
        data_clean = data[feature_columns + ['Future_return']].dropna()
        
        return data_clean[feature_columns], data_clean['Future_return']
    
    def train_ml_models(self, symbol, data):
        """Entraîne les modèles de machine learning"""
        print(f"\n🤖 Entraînement des modèles ML pour {symbol}...")
        
        X, y = self.prepare_ml_features(data)
        
        if len(X) < 100:
            print("❌ Pas assez de données pour l'entraînement ML")
            return False
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Normalisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[symbol] = scaler
        
        # Modèles
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        self.models[symbol] = {}
        model_scores = {}
        
        for name, model in models.items():
            if name == 'LinearRegression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            score = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            self.models[symbol][name] = model
            model_scores[name] = {'R2': score, 'MSE': mse}
            
            print(f"  {name}: R² = {score:.4f}, MSE = {mse:.6f}")
        
        # Sélection du meilleur modèle
        best_model = max(model_scores.keys(), key=lambda x: model_scores[x]['R2'])
        print(f"✅ Meilleur modèle: {best_model}")
        
        return True
    
    def get_ml_signal(self, symbol, current_data):
        """Obtient un signal de trading basé sur ML"""
        if symbol not in self.models or not self.models[symbol]:
            return 0, 0.5  # Signal neutre si pas de modèle
        
        try:
            # Préparation des features actuelles
            X_current, _ = self.prepare_ml_features(current_data.tail(100))
            
            if len(X_current) == 0:
                return 0, 0.5
            
            X_latest = X_current.iloc[-1:].values
            
            # Prédictions de tous les modèles
            predictions = []
            for name, model in self.models[symbol].items():
                if name == 'LinearRegression':
                    X_scaled = self.scalers[symbol].transform(X_latest)
                    pred = model.predict(X_scaled)[0]
                else:
                    pred = model.predict(X_latest)[0]
                predictions.append(pred)
            
            # Prédiction moyenne
            avg_prediction = np.mean(predictions)
            confidence = 1 - np.std(predictions)  # Confiance basée sur consensus
            
            # Signal basé sur la prédiction
            if avg_prediction > 0.005:  # +0.5%
                signal = 1  # Achat
            elif avg_prediction < -0.005:  # -0.5%
                signal = -1  # Vente
            else:
                signal = 0  # Neutre
            
            return signal, min(max(confidence, 0), 1)
            
        except Exception as e:
            print(f"⚠️ Erreur ML signal: {e}")
            return 0, 0.5
    
    def multi_timeframe_analysis(self, symbol):
        """Analyse multi-timeframes"""
        timeframes = {
            '1h': {'period': '30d', 'interval': '1h', 'weight': 0.2},
            '4h': {'period': '60d', 'interval': '4h', 'weight': 0.3},
            '1d': {'period': '1y', 'interval': '1d', 'weight': 0.5}
        }
        
        signals = {}
        
        for tf, params in timeframes.items():
            data = self.fetch_data(symbol, params['period'], params['interval'])
            if data is not None and len(data) > 50:
                signal = self.get_combined_signal(data)
                signals[tf] = {
                    'signal': signal['signal'],
                    'strength': signal['strength'],
                    'weight': params['weight']
                }
        
        if not signals:
            return {'signal': 0, 'strength': 0.5, 'timeframes': {}}
        
        # Signal pondéré
        weighted_signal = sum(s['signal'] * s['weight'] for s in signals.values())
        weighted_strength = sum(s['strength'] * s['weight'] for s in signals.values())
        
        final_signal = 1 if weighted_signal > 0.3 else (-1 if weighted_signal < -0.3 else 0)
        
        return {
            'signal': final_signal,
            'strength': weighted_strength,
            'timeframes': signals,
            'weighted_signal': weighted_signal
        }
    
    def get_combined_signal(self, data):
        """Combine plusieurs signaux pour une décision finale"""
        if len(data) < 50:
            return {'signal': 0, 'strength': 0.5}
        
        signals = []
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Signal 1: Croisement moyennes mobiles
        if current['SMA_20'] > current['SMA_50'] and prev['SMA_20'] <= prev['SMA_50']:
            signals.append(1)  # Golden cross
        elif current['SMA_20'] < current['SMA_50'] and prev['SMA_20'] >= prev['SMA_50']:
            signals.append(-1)  # Death cross
        
        # Signal 2: RSI
        if current['RSI'] < 30:
            signals.append(1)  # Survente
        elif current['RSI'] > 70:
            signals.append(-1)  # Surachat
        
        # Signal 3: MACD
        if current['MACD'] > current['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
            signals.append(1)  # Croisement haussier
        elif current['MACD'] < current['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
            signals.append(-1)  # Croisement baissier
        
        # Signal 4: Bollinger Bands
        if current['Close'] < current['BB_lower']:
            signals.append(1)  # Rebond potentiel
        elif current['Close'] > current['BB_upper']:
            signals.append(-1)  # Correction potentielle
        
        # Signal 5: Stochastic
        if current['Stoch_K'] < 20 and current['Stoch_K'] > current['Stoch_D']:
            signals.append(1)
        elif current['Stoch_K'] > 80 and current['Stoch_K'] < current['Stoch_D']:
            signals.append(-1)
        
        if not signals:
            return {'signal': 0, 'strength': 0.5}
        
        # Calcul du signal final
        avg_signal = np.mean(signals)
        strength = abs(avg_signal)
        
        final_signal = 1 if avg_signal > 0.2 else (-1 if avg_signal < -0.2 else 0)
        
        return {'signal': final_signal, 'strength': min(strength, 1.0)}
    
    def backtest_strategy(self, symbol, data, strategy_name="Combined", 
                         initial_capital=10000, commission=0.001):
        """Backtesting complet d'une stratégie"""
        print(f"\n📊 Backtesting de la stratégie {strategy_name} pour {symbol}")
        
        if len(data) < 100:
            print("❌ Pas assez de données pour le backtesting")
            return None
        
        # Initialisation
        capital = initial_capital
        position = 0  # 0: neutre, 1: long, -1: short
        entry_price = 0
        trades = []
        equity_curve = []
        
        for i in range(50, len(data)):
            current_data = data.iloc[:i+1]
            current_row = data.iloc[i]
            
            # Obtenir le signal
            signal_data = self.get_combined_signal(current_data)
            signal = signal_data['signal']
            strength = signal_data['strength']
            
            date = current_row.name
            price = current_row['Close']
            
            # Gestion des positions
            if position == 0 and signal != 0 and strength > 0.6:
                # Entrée en position
                position = signal
                entry_price = price
                trade_capital = capital * 0.95  # 95% du capital
                shares = trade_capital / price
                commission_cost = trade_capital * commission
                capital -= commission_cost
                
                trades.append({
                    'date': date,
                    'type': 'ENTRY',
                    'signal': signal,
                    'price': price,
                    'shares': shares,
                    'capital': capital,
                    'strength': strength
                })
                
            elif position != 0:
                # Gestion de la sortie
                exit_signal = False
                
                # Stop loss et take profit
                if position == 1:  # Position longue
                    pnl_pct = (price - entry_price) / entry_price
                    if pnl_pct <= -0.05 or pnl_pct >= 0.15:  # -5% stop loss, +15% take profit
                        exit_signal = True
                    elif signal == -1 and strength > 0.6:  # Signal contraire fort
                        exit_signal = True
                        
                elif position == -1:  # Position courte
                    pnl_pct = (entry_price - price) / entry_price
                    if pnl_pct <= -0.05 or pnl_pct >= 0.15:
                        exit_signal = True
                    elif signal == 1 and strength > 0.6:
                        exit_signal = True
                
                if exit_signal:
                    # Sortie de position
                    trade_value = shares * price
                    commission_cost = trade_value * commission
                    capital = trade_value - commission_cost
                    
                    pnl = capital - initial_capital if len(trades) == 1 else capital - trades[-1]['capital']
                    
                    trades.append({
                        'date': date,
                        'type': 'EXIT',
                        'signal': -position,
                        'price': price,
                        'shares': shares,
                        'capital': capital,
                        'pnl': pnl,
                        'pnl_pct': pnl / entry_price if entry_price > 0 else 0
                    })
                    
                    position = 0
                    entry_price = 0
            
            # Courbe de capital
            if position == 0:
                current_capital = capital
            else:
                current_value = shares * price
                current_capital = current_value - (shares * entry_price - capital)
            
            equity_curve.append({
                'date': date,
                'equity': current_capital,
                'price': price
            })
        
        # Calcul des métriques de performance
        if len(trades) < 2:
            print("❌ Pas assez de trades pour l'analyse")
            return None
        
        metrics = self.calculate_performance_metrics(trades, equity_curve, initial_capital)
        
        self.backtest_results[f"{symbol}_{strategy_name}"] = {
            'trades': trades,
            'equity_curve': equity_curve,
            'metrics': metrics
        }
        
        self.print_backtest_results(symbol, strategy_name, metrics)
        return metrics
    
    def calculate_performance_metrics(self, trades, equity_curve, initial_capital):
        """Calcule les métriques de performance détaillées"""
        # Extraction des PnL
        completed_trades = [t for t in trades if t['type'] == 'EXIT']
        
        if not completed_trades:
            return {}
        
        pnls = [trade['pnl'] for trade in completed_trades]
        
        # Métriques de base
        total_trades = len(completed_trades)
        winning_trades = len([p for p in pnls if p > 0])
        losing_trades = len([p for p in pnls if p < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_return = (equity_curve[-1]['equity'] - initial_capital) / initial_capital
        avg_return_per_trade = np.mean(pnls) / initial_capital if pnls else 0
        
        # Drawdown
        peak = initial_capital
        max_drawdown = 0
        drawdowns = []
        
        for point in equity_curve:
            if point['equity'] > peak:
                peak = point['equity']
            drawdown = (peak - point['equity']) / peak
            drawdowns.append(drawdown)
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Ratio de Sharpe (approximatif)
        if len(pnls) > 1:
            returns_std = np.std(pnls) / initial_capital
            sharpe_ratio = avg_return_per_trade / returns_std if returns_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Profit factor
        gross_profit = sum([p for p in pnls if p > 0])
        gross_loss = abs(sum([p for p in pnls if p < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'avg_return_per_trade': avg_return_per_trade,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'final_capital': equity_curve[-1]['equity']
        }
    
    def print_backtest_results(self, symbol, strategy, metrics):
        """Affiche les résultats du backtesting"""
        print(f"\n{'='*60}")
        print(f"📈 RÉSULTATS BACKTESTING - {symbol} ({strategy})")
        print(f"{'='*60}")
        
        print(f"💰 Performance Globale:")
        print(f"   Rendement Total: {metrics['total_return_pct']:.2f}%")
        print(f"   Capital Final: ${metrics['final_capital']:,.2f}")
        print(f"   Drawdown Max: {metrics['max_drawdown_pct']:.2f}%")
        
        print(f"\n📊 Statistiques de Trading:")
        print(f"   Nombre de Trades: {metrics['total_trades']}")
        print(f"   Trades Gagnants: {metrics['winning_trades']}")
        print(f"   Trades Perdants: {metrics['losing_trades']}")
        print(f"   Taux de Réussite: {metrics['win_rate']*100:.1f}%")
        
        print(f"\n📏 Métriques de Risque:")
        print(f"   Ratio de Sharpe: {metrics['sharpe_ratio']:.2f}")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   Profit Brut: ${metrics['gross_profit']:,.2f}")
        print(f"   Perte Brute: ${metrics['gross_loss']:,.2f}")
    
    def optimize_strategy_parameters(self, symbol, data):
        """Optimise automatiquement les paramètres de stratégie"""
        print(f"\n🔧 Optimisation des paramètres pour {symbol}...")
        
        # Paramètres à optimiser
        rsi_thresholds = [(20, 80), (25, 75), (30, 70)]
        sma_periods = [(10, 30), (20, 50), (15, 45)]
        stop_loss_levels = [0.03, 0.05, 0.07]  # 3%, 5%, 7%
        take_profit_levels = [0.10, 0.15, 0.20]  # 10%, 15%, 20%
        
        best_params = None
        best_performance = -float('inf')
        
        optimization_results = []
        
        for rsi_low, rsi_high in rsi_thresholds:
            for sma_short, sma_long in sma_periods:
                for stop_loss in stop_loss_levels:
                    for take_profit in take_profit_levels:
                        
                        # Simulation avec ces paramètres
                        performance = self.simulate_with_parameters(
                            data, rsi_low, rsi_high, sma_short, sma_long, 
                            stop_loss, take_profit
                        )
                        
                        optimization_results.append({
                            'params': {
                                'rsi_low': rsi_low,
                                'rsi_high': rsi_high,
                                'sma_short': sma_short,
                                'sma_long': sma_long,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit
                            },
                            'performance': performance,
                            'score': performance.get('total_return', 0) - performance.get('max_drawdown', 1)
                        })
        
        # Sélection des meilleurs paramètres
        if optimization_results:
            best_result = max(optimization_results, key=lambda x: x['score'])
            best_params = best_result['params']
            best_performance = best_result['performance']
            
            print(f"✅ Meilleurs paramètres trouvés:")
            for param, value in best_params.items():
                print(f"   {param}: {value}")
            print(f"   Score: {best_result['score']:.4f}")
            print(f"   Rendement: {best_performance.get('total_return_pct', 0):.2f}%")
        
        return best_params, optimization_results
    
    def simulate_with_parameters(self, data, rsi_low, rsi_high, sma_short, sma_long, 
                                stop_loss, take_profit, initial_capital=10000):
        """Simule une stratégie avec des paramètres spécifiques"""
        if len(data) < max(sma_long, 50):
            return {'total_return': -1, 'max_drawdown': 1}
        
        # Recalcul des indicateurs avec nouveaux paramètres
        data_copy = data.copy()
        data_copy[f'SMA_{sma_short}'] = ta.trend.sma_indicator(data_copy['Close'], window=sma_short)
        data_copy[f'SMA_{sma_long}'] = ta.trend.sma_indicator(data_copy['Close'], window=sma_long)
        
        capital = initial_capital
        position = 0
        entry_price = 0
        equity_curve = []
        
        for i in range(sma_long, len(data_copy)):
            current = data_copy.iloc[i]
            prev = data_copy.iloc[i-1]
            
            price = current['Close']
            
            # Signaux avec paramètres optimisés
            signal = 0
            
            # Croisement moyennes mobiles
            if (current[f'SMA_{sma_short}'] > current[f'SMA_{sma_long}'] and 
                prev[f'SMA_{sma_short}'] <= prev[f'SMA_{sma_long}']):
                signal = 1
            elif (current[f'SMA_{sma_short}'] < current[f'SMA_{sma_long}'] and 
                  prev[f'SMA_{sma_short}'] >= prev[f'SMA_{sma_long}']):
                signal = -1
            
            # Confirmation RSI
            if signal == 1 and current['RSI'] > rsi_high:
                signal = 0  # Pas d'achat si surachat
            elif signal == -1 and current['RSI'] < rsi_low:
                signal = 0  # Pas de vente si survente
            
            # Gestion des positions
            if position == 0 and signal != 0:
                position = signal
                entry_price = price
                shares = capital * 0.95 / price
                
            elif position != 0:
                pnl_pct = (price - entry_price) / entry_price * position
                
                if pnl_pct <= -stop_loss or pnl_pct >= take_profit or signal == -position:
                    capital = shares * price * 0.999  # Commission
                    position = 0
                    entry_price = 0
            
            # Équité courante
            if position == 0:
                current_equity = capital
            else:
                current_equity = shares * price * 0.999
            
            equity_curve.append(current_equity)
        
        if not equity_curve:
            return {'total_return': -1, 'max_drawdown': 1}
        
        # Calcul des métriques simplifiées
        final_equity = equity_curve[-1]
        total_return = (final_equity - initial_capital) / initial_capital
        
        # Drawdown
        peak = initial_capital
        max_drawdown = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'max_drawdown': max_drawdown,
            'final_equity': final_equity
        }
    
    def generate_trading_report(self, symbol):
        """Génère un rapport de trading complet"""
        print(f"\n📋 RAPPORT DE TRADING COMPLET - {symbol}")
        print("="*70)
        
        # Analyse multi-timeframes
        mtf_analysis = self.multi_timeframe_analysis(symbol)
        
        print(f"\n🔍 ANALYSE MULTI-TIMEFRAMES:")
        print(f"Signal Global: {self.signal_to_text(mtf_analysis['signal'])}")
        print(f"Force du Signal: {mtf_analysis['strength']:.2f}")
        print(f"Score Pondéré: {mtf_analysis.get('weighted_signal', 0):.2f}")
        
        for tf, data in mtf_analysis.get('timeframes', {}).items():
            print(f"  {tf}: {self.signal_to_text(data['signal'])} (Force: {data['strength']:.2f})")
        
        # Analyse ML si disponible
        if symbol in self.models and self.models[symbol]:
            daily_data = self.fetch_data(symbol, period="1y", interval="1d")
            if daily_data is not None:
                ml_signal, ml_confidence = self.get_ml_signal(symbol, daily_data)
                print(f"\n🤖 ANALYSE MACHINE LEARNING:")
                print(f"Signal ML: {self.signal_to_text(ml_signal)}")
                print(f"Confiance: {ml_confidence:.2f}")
                
                # Prédictions individuelles des modèles
                X, _ = self.prepare_ml_features(daily_data.tail(100))
                if len(X) > 0:
                    X_latest = X.iloc[-1:].values
                    print(f"Prédictions des modèles:")
                    for name, model in self.models[symbol].items():
                        try:
                            if name == 'LinearRegression':
                                X_scaled = self.scalers[symbol].transform(X_latest)
                                pred = model.predict(X_scaled)[0]
                            else:
                                pred = model.predict(X_latest)[0]
                            print(f"  {name}: {pred*100:.2f}% de rendement prédit")
                        except:
                            pass
        
        # Recommandations finales
        print(f"\n💡 RECOMMANDATIONS:")
        overall_signal = mtf_analysis['signal']
        overall_strength = mtf_analysis['strength']
        
        if overall_signal == 1 and overall_strength > 0.7:
            print("🟢 FORTE RECOMMANDATION D'ACHAT")
            print("   - Signaux haussiers convergents sur plusieurs timeframes")
            print("   - Niveau de confiance élevé")
        elif overall_signal == 1 and overall_strength > 0.5:
            print("🟡 RECOMMANDATION D'ACHAT MODÉRÉE")
            print("   - Signaux haussiers présents mais force modérée")
            print("   - Surveiller les confirmations")
        elif overall_signal == -1 and overall_strength > 0.7:
            print("🔴 FORTE RECOMMANDATION DE VENTE")
            print("   - Signaux baissiers convergents")
            print("   - Risque de correction important")
        elif overall_signal == -1 and overall_strength > 0.5:
            print("🟡 RECOMMANDATION DE VENTE MODÉRÉE")
            print("   - Signaux baissiers présents")
            print("   - Prudence recommandée")
        else:
            print("⚪ POSITION NEUTRE")
            print("   - Signaux mixtes ou faibles")
            print("   - Attendre des signaux plus clairs")
        
        return {
            'symbol': symbol,
            'multi_timeframe': mtf_analysis,
            'ml_analysis': {'signal': ml_signal, 'confidence': ml_confidence} if symbol in self.models else None,
            'recommendation': overall_signal,
            'strength': overall_strength
        }
    
    def signal_to_text(self, signal):
        """Convertit un signal numérique en texte"""
        if signal == 1:
            return "🟢 ACHAT"
        elif signal == -1:
            return "🔴 VENTE"
        else:
            return "⚪ NEUTRE"
    
    def auto_trade_execution(self, symbol, signal_data, position_size=0.1):
        """Exécution automatique des trades (simulation)"""
        print(f"\n🚀 EXÉCUTION AUTOMATIQUE - {symbol}")
        
        signal = signal_data['signal']
        strength = signal_data['strength']
        
        if signal == 0 or strength < 0.6:
            print("❌ Signal trop faible pour l'exécution automatique")
            return False
        
        # Récupération du prix actuel
        current_data = self.fetch_data(symbol, period="5d", interval="1h")
        if current_data is None or len(current_data) == 0:
            print("❌ Impossible de récupérer le prix actuel")
            return False
        
        current_price = current_data['Close'].iloc[-1]
        
        # Calcul de la taille de position
        risk_per_trade = 0.02  # 2% du capital par trade
        stop_loss_pct = 0.05  # Stop loss à 5%
        
        # Simulation d'exécution
        if symbol not in self.active_positions:
            self.active_positions[symbol] = []
        
        trade = {
            'timestamp': datetime.now(),
            'signal': signal,
            'entry_price': current_price,
            'position_size': position_size,
            'stop_loss': current_price * (1 - stop_loss_pct) if signal == 1 else current_price * (1 + stop_loss_pct),
            'take_profit': current_price * (1 + 0.15) if signal == 1 else current_price * (1 - 0.15),
            'strength': strength,
            'status': 'ACTIVE'
        }
        
        self.active_positions[symbol].append(trade)
        
        print(f"✅ Trade exécuté:")
        print(f"   Signal: {self.signal_to_text(signal)}")
        print(f"   Prix d'entrée: ${current_price:.2f}")
        print(f"   Stop Loss: ${trade['stop_loss']:.2f}")
        print(f"   Take Profit: ${trade['take_profit']:.2f}")
        print(f"   Force du signal: {strength:.2f}")
        
        return True
    
    def monitor_active_positions(self):
        """Surveillance des positions actives"""
        if not self.active_positions:
            print("📊 Aucune position active à surveiller")
            return
        
        print(f"\n👁️ SURVEILLANCE DES POSITIONS ACTIVES")
        print("="*50)
        
        for symbol, positions in self.active_positions.items():
            active_positions = [p for p in positions if p['status'] == 'ACTIVE']
            
            if not active_positions:
                continue
                
            print(f"\n📈 {symbol}:")
            
            # Prix actuel
            current_data = self.fetch_data(symbol, period="2d", interval="1h")
            if current_data is None:
                continue
                
            current_price = current_data['Close'].iloc[-1]
            
            for i, position in enumerate(active_positions):
                entry_price = position['entry_price']
                signal = position['signal']
                
                # Calcul du PnL
                if signal == 1:  # Position longue
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:  # Position courte
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                
                pnl_color = "🟢" if pnl_pct > 0 else "🔴"
                
                print(f"  Position #{i+1}:")
                print(f"    Signal: {self.signal_to_text(signal)}")
                print(f"    Prix d'entrée: ${entry_price:.2f}")
                print(f"    Prix actuel: ${current_price:.2f}")
                print(f"    PnL: {pnl_color} {pnl_pct:+.2f}%")
                print(f"    Stop Loss: ${position['stop_loss']:.2f}")
                print(f"    Take Profit: ${position['take_profit']:.2f}")
                
                # Vérification des conditions de sortie
                if signal == 1:  # Position longue
                    if current_price <= position['stop_loss']:
                        print(f"    ⚠️ STOP LOSS ATTEINT!")
                        position['status'] = 'CLOSED'
                        position['exit_reason'] = 'STOP_LOSS'
                    elif current_price >= position['take_profit']:
                        print(f"    🎯 TAKE PROFIT ATTEINT!")
                        position['status'] = 'CLOSED'
                        position['exit_reason'] = 'TAKE_PROFIT'
                else:  # Position courte
                    if current_price >= position['stop_loss']:
                        print(f"    ⚠️ STOP LOSS ATTEINT!")
                        position['status'] = 'CLOSED'
                        position['exit_reason'] = 'STOP_LOSS'
                    elif current_price <= position['take_profit']:
                        print(f"    🎯 TAKE PROFIT ATTEINT!")
                        position['status'] = 'CLOSED'
                        position['exit_reason'] = 'TAKE_PROFIT'
                
                print()
    
    def run_complete_analysis(self, symbol):
        """Lance une analyse complète avec toutes les fonctionnalités"""
        print(f"\n🎯 ANALYSE COMPLÈTE - {symbol}")
        print("="*60)
        
        # 1. Récupération des données
        data = self.fetch_data(symbol, period="2y", interval="1d")
        if data is None:
            return None
        
        # 2. Entraînement des modèles ML
        ml_success = self.train_ml_models(symbol, data)
        
        # 3. Backtesting
        backtest_results = self.backtest_strategy(symbol, data)
        
        # 4. Optimisation des paramètres
        if len(data) > 200:  # Seulement si assez de données
            best_params, optimization_results = self.optimize_strategy_parameters(symbol, data)
        
        # 5. Rapport de trading
        trading_report = self.generate_trading_report(symbol)
        
        # 6. Recommandation finale
        final_recommendation = self.get_final_recommendation(symbol, trading_report, backtest_results)
        
        return {
            'symbol': symbol,
            'ml_trained': ml_success,
            'backtest': backtest_results,
            'trading_report': trading_report,
            'recommendation': final_recommendation
        }
    
    def get_final_recommendation(self, symbol, trading_report, backtest_results):
        """Formule une recommandation finale basée sur toutes les analyses"""
        print(f"\n🎯 RECOMMANDATION FINALE - {symbol}")
        print("="*50)
        
        score = 0
        factors = []
        
        # Facteur 1: Signal multi-timeframes
        mtf_signal = trading_report['multi_timeframe']['signal']
        mtf_strength = trading_report['multi_timeframe']['strength']
        
        if mtf_signal == 1 and mtf_strength > 0.7:
            score += 3
            factors.append("✅ Signaux haussiers forts multi-timeframes")
        elif mtf_signal == 1 and mtf_strength > 0.5:
            score += 1
            factors.append("🟡 Signaux haussiers modérés")
        elif mtf_signal == -1 and mtf_strength > 0.7:
            score -= 3
            factors.append("❌ Signaux baissiers forts")
        elif mtf_signal == -1 and mtf_strength > 0.5:
            score -= 1
            factors.append("🟡 Signaux baissiers modérés")
        
        # Facteur 2: Performance du backtesting
        if backtest_results:
            win_rate = backtest_results.get('win_rate', 0)
            total_return = backtest_results.get('total_return', 0)
            max_drawdown = backtest_results.get('max_drawdown', 1)
            
            if win_rate > 0.6 and total_return > 0.15 and max_drawdown < 0.15:
                score += 2
                factors.append("✅ Excellent historique de performance")
            elif win_rate > 0.5 and total_return > 0:
                score += 1
                factors.append("🟡 Performance historique correcte")
            else:
                score -= 1
                factors.append("❌ Performance historique faible")
        
        # Facteur 3: ML si disponible
        if trading_report.get('ml_analysis'):
            ml_signal = trading_report['ml_analysis']['signal']
            ml_confidence = trading_report['ml_analysis']['confidence']
            
            if ml_signal == mtf_signal and ml_confidence > 0.7:
                score += 1
                factors.append("✅ Confirmation par IA")
            elif ml_signal != mtf_signal:
                score -= 1
                factors.append("⚠️ Divergence avec l'IA")
        
        # Recommandation finale
        if score >= 4:
            recommendation = "FORTE RECOMMANDATION D'ACHAT"
            action = "🟢 ACHETER"
            risk_level = "Faible à Modéré"
        elif score >= 2:
            recommendation = "RECOMMANDATION D'ACHAT"
            action = "🟡 ACHETER (avec prudence)"
            risk_level = "Modéré"
        elif score <= -4:
            recommendation = "FORTE RECOMMANDATION DE VENTE"
            action = "🔴 VENDRE"
            risk_level = "Élevé"
        elif score <= -2:
            recommendation = "RECOMMANDATION DE VENTE"
            action = "🟡 VENDRE (avec prudence)"
            risk_level = "Modéré à Élevé"
        else:
            recommendation = "POSITION NEUTRE"
            action = "⚪ ATTENDRE"
            risk_level = "Variable"
        
        print(f"Action Recommandée: {action}")
        print(f"Niveau de Risque: {risk_level}")
        print(f"Score Global: {score}/10")
        print(f"\nFacteurs Analysés:")
        for factor in factors:
            print(f"  {factor}")
        
        print(f"\n💼 CONSEILS DE GESTION:")
        if score > 0:
            print("  • Taille de position recommandée: 2-5% du portefeuille")
            print("  • Stop loss suggéré: 5-7% sous le prix d'entrée")
            print("  • Take profit: 15-20% au-dessus du prix d'entrée")
        elif score < 0:
            print("  • Éviter les nouvelles positions longues")
            print("  • Considérer une position courte si expertise suffisante")
            print("  • Surveiller les signaux de retournement")
        else:
            print("  • Attendre des signaux plus clairs")
            print("  • Surveiller l'évolution des indicateurs")
            print("  • Maintenir la diversification")
        
        return {
            'action': action,
            'recommendation': recommendation,
            'score': score,
            'risk_level': risk_level,
            'factors': factors
        }


# Exemple d'utilisation du système complet
def demo_slm_algorithmic_trading():
    """Démonstration du système de trading algorithmique SLM"""
    print("🚀 DÉMONSTRATION SLM ALGORITHMIC TRADER")
    print("="*60)
    
    # Initialisation
    trader = SLMAlgorithmicTrader()
    
    # Symboles à analyser
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    print("📊 Analyse en cours des symboles:", symbols)
    print("⏱️ Cela peut prendre quelques minutes...")
    
    results = {}
    
    for symbol in symbols:
        print(f"\n{'='*20} {symbol} {'='*20}")
        
        try:
            # Analyse complète
            result = trader.run_complete_analysis(symbol)
            results[symbol] = result
            
            # Simulation d'exécution automatique si signal fort
            if result and result['recommendation']['score'] >= 3:
                trader.auto_trade_execution(symbol, result['trading_report']['multi_timeframe'])
            
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse de {symbol}: {e}")
            continue
    
    # Surveillance des positions
    trader.monitor_active_positions()
    
    # Résumé final
    print(f"\n📋 RÉSUMÉ FINAL")
    print("="*40)
    
    for symbol, result in results.items():
        if result:
            rec = result['recommendation']
            print(f"{symbol}: {rec['action']} (Score: {rec['score']})")
    
    return trader, results


if __name__ == "__main__":
    # Démonstration
    trader, results = demo_slm_algorithmic_trading()
    
    print("\n🎉 Démonstration terminée!")
    print("\nFonctionnalités implémentées:")
    print("✅ Machine Learning multi-modèles")
    print("✅ Backtesting complet avec métriques")
    print("✅ Optimisation automatique des paramètres")
    print("✅ Analyse multi-timeframes")
    print("✅ Exécution automatique simulée")
    print("✅ Surveillance des positions")
    print("✅ Recommandations basées sur l'IA")
    
    print(f"\n💡 Le Module 4 de SLM TRADE est maintenant opérationnel!")
    print("Prêt pour l'intégration avec les autres modules du système.")