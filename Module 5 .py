"""
SLM TRADE - Module 5: Gestion Avancée des Risques
==================================================

Ce module implémente un système complet de gestion des risques avec:
- Value at Risk (VaR) et Expected Shortfall (ES)
- Position sizing dynamique avec Kelly Criterion
- Gestion de portefeuille multi-actifs
- Corrélations et diversification
- Stress testing et Monte Carlo
- Alertes de risque en temps réel
- Dashboard de visualisation des risques

Auteur: SLM TRADE System
Version: 5.0
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SLMRiskManager:
    """Gestionnaire de risques avancé pour SLM TRADE"""
    
    def __init__(self):
        """Initialisation du gestionnaire de risques"""
        
        # Paramètres de risque
        self.max_portfolio_var = 0.02  # VaR max 2% du capital
        self.max_position_weight = 0.20  # 20% max par position
        self.confidence_levels = [0.90, 0.95, 0.99]
        self.lookback_days = 252  # 1 an pour calculs historiques
        
        # Capital et portfolio
        self.total_capital = 100000
        self.available_capital = 100000
        self.portfolio_positions = {}
        self.portfolio_weights = {}
        
        # Cache des données
        self.price_data = {}
        self.returns_data = {}
        self.correlation_matrix = None
        self.last_update = None
        
        print("🛡️ SLM Risk Manager initialisé avec succès!")
    
    def set_capital(self, capital):
        """Définir le capital total"""
        self.total_capital = capital
        self.available_capital = capital
        print(f"💰 Capital défini: ${capital:,.2f}")
    
    def fetch_risk_data(self, symbols, period="1y"):
        """Récupération des données pour analyse de risque"""
        
        print(f"📊 Récupération données risque pour {len(symbols)} actifs...")
        
        try:
            for symbol in symbols:
                # Télécharger les données
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if len(data) > 20:
                    self.price_data[symbol] = data['Close']
                    # Calcul des rendements
                    returns = data['Close'].pct_change().dropna()
                    self.returns_data[symbol] = returns
            
            # Calculer matrice de corrélation
            if len(self.returns_data) > 1:
                returns_df = pd.DataFrame(self.returns_data)
                self.correlation_matrix = returns_df.corr()
            
            self.last_update = datetime.now()
            print(f"✅ Données mises à jour: {len(self.price_data)} actifs")
            
        except Exception as e:
            print(f"❌ Erreur récupération données: {e}")
    
    def calculate_var(self, returns, confidence_level=0.95, method='historical'):
        """Calcul du Value at Risk (VaR)"""
        
        if len(returns) < 30:
            return 0
        
        if method == 'historical':
            # VaR historique
            var = np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            # VaR paramétrique (distribution normale)
            mean_return = returns.mean()
            std_return = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean_return + z_score * std_return
        
        elif method == 'monte_carlo':
            # VaR Monte Carlo
            simulated_returns = np.random.normal(
                returns.mean(), returns.std(), 10000
            )
            var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        
        return abs(var)
    
    def calculate_expected_shortfall(self, returns, confidence_level=0.95):
        """Calcul de l'Expected Shortfall (ES)"""
        
        var = self.calculate_var(returns, confidence_level)
        # ES = moyenne des pertes au-delà du VaR
        tail_losses = returns[returns <= -var]
        
        if len(tail_losses) > 0:
            es = abs(tail_losses.mean())
        else:
            es = var
        
        return es
    
    def calculate_position_size_kelly(self, symbol, win_prob, avg_win, avg_loss):
        """Position sizing avec Kelly Criterion"""
        
        if avg_loss == 0 or win_prob <= 0:
            return 0
        
        # Formule Kelly: f = (bp - q) / b
        # b = avg_win/avg_loss, p = win_prob, q = 1-win_prob
        b = avg_win / abs(avg_loss)
        p = win_prob
        q = 1 - win_prob
        
        kelly_fraction = (b * p - q) / b
        
        # Limitation à 25% pour sécurité
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        
        # Calcul taille position
        position_value = self.available_capital * kelly_fraction
        
        return {
            'kelly_fraction': kelly_fraction,
            'position_value': position_value,
            'recommended_allocation': kelly_fraction * 100
        }
    
    def calculate_optimal_position_size(self, symbol, target_volatility=0.15):
        """Calcul taille position optimale basée sur volatilité cible"""
        
        if symbol not in self.returns_data:
            return {'position_value': 0, 'shares': 0, 'weight': 0}
        
        returns = self.returns_data[symbol]
        asset_volatility = returns.std() * np.sqrt(252)  # Volatilité annualisée
        
        if asset_volatility == 0:
            return {'position_value': 0, 'shares': 0, 'weight': 0}
        
        # Position sizing basé sur volatilité
        position_weight = target_volatility / asset_volatility
        position_weight = min(position_weight, self.max_position_weight)
        
        position_value = self.available_capital * position_weight
        
        # Calcul nombre d'actions
        if symbol in self.price_data:
            current_price = self.price_data[symbol].iloc[-1]
            shares = int(position_value / current_price)
        else:
            shares = 0
        
        return {
            'position_value': position_value,
            'shares': shares,
            'weight': position_weight,
            'asset_volatility': asset_volatility
        }
    
    def analyze_portfolio_risk(self):
        """Analyse complète du risque portefeuille"""
        
        if not self.portfolio_positions:
            return {"error": "Aucune position dans le portefeuille"}
        
        # Calculs de base
        portfolio_value = sum(self.portfolio_positions.values())
        weights = np.array([pos/portfolio_value for pos in self.portfolio_positions.values()])
        symbols = list(self.portfolio_positions.keys())
        
        # VaR individuel de chaque position
        individual_vars = {}
        for symbol in symbols:
            if symbol in self.returns_data:
                returns = self.returns_data[symbol]
                var_95 = self.calculate_var(returns, 0.95)
                position_var = self.portfolio_positions[symbol] * var_95
                individual_vars[symbol] = position_var
        
        # VaR portefeuille (en tenant compte des corrélations)
        portfolio_var = self.calculate_portfolio_var(symbols, weights)
        
        # Expected Shortfall portefeuille
        portfolio_es = portfolio_var * 1.3  # Approximation ES ≈ 1.3 * VaR
        
        # Bénéfice de diversification
        sum_individual_vars = sum(individual_vars.values())
        diversification_benefit = (sum_individual_vars - portfolio_var) / sum_individual_vars if sum_individual_vars > 0 else 0
        
        # Concentration du portefeuille (indice Herfindahl)
        herfindahl_index = sum(w**2 for w in weights)
        effective_positions = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        return {
            'portfolio_value': portfolio_value,
            'portfolio_var_95': portfolio_var,
            'portfolio_es_95': portfolio_es,
            'var_percentage': (portfolio_var / portfolio_value) * 100,
            'individual_vars': individual_vars,
            'diversification_benefit': diversification_benefit * 100,
            'concentration_index': herfindahl_index,
            'effective_positions': effective_positions,
            'largest_position_weight': max(weights) * 100
        }
    
    def calculate_portfolio_var(self, symbols, weights):
        """Calcul VaR portefeuille avec corrélations"""
        
        if len(symbols) < 2 or self.correlation_matrix is None:
            return 0
        
        # Matrice de covariance des rendements
        returns_matrix = pd.DataFrame({symbol: self.returns_data[symbol] 
                                     for symbol in symbols if symbol in self.returns_data})
        
        if returns_matrix.empty:
            return 0
        
        cov_matrix = returns_matrix.cov() * 252  # Annualisée
        
        # VaR portefeuille = sqrt(w' * Σ * w) * z_score
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Z-score pour 95% de confiance
        z_score = stats.norm.ppf(0.95)
        portfolio_var = portfolio_volatility * z_score
        
        return portfolio_var * sum(self.portfolio_positions.values())
    
    def stress_test_portfolio(self, scenarios=None):
        """Tests de stress sur le portefeuille"""
        
        if not scenarios:
            scenarios = {
                'Crash 2008': {'market_drop': -0.37, 'vol_increase': 2.5},
                'COVID Mars 2020': {'market_drop': -0.34, 'vol_increase': 3.0},
                'Dot-com 2000': {'market_drop': -0.49, 'vol_increase': 2.0},
                'Crise personnalisée': {'market_drop': -0.25, 'vol_increase': 2.0}
            }
        
        stress_results = {}
        current_portfolio_value = sum(self.portfolio_positions.values())
        
        for scenario_name, params in scenarios.items():
            scenario_loss = 0
            
            for symbol, position_value in self.portfolio_positions.items():
                if symbol in self.returns_data:
                    # Simuler l'impact du scénario
                    base_loss = position_value * abs(params['market_drop'])
                    
                    # Ajuster selon la volatilité de l'actif
                    returns = self.returns_data[symbol]
                    asset_vol = returns.std() * np.sqrt(252)
                    vol_adjustment = asset_vol * params['vol_increase']
                    
                    total_loss = base_loss * (1 + vol_adjustment)
                    scenario_loss += total_loss
            
            loss_percentage = (scenario_loss / current_portfolio_value) * 100
            
            stress_results[scenario_name] = {
                'loss_amount': scenario_loss,
                'loss_percentage': loss_percentage,
                'remaining_capital': max(0, current_portfolio_value - scenario_loss)
            }
        
        return stress_results
    
    def generate_risk_alerts(self):
        """Génération d'alertes de risque"""
        
        alerts = []
        
        if not self.portfolio_positions:
            return alerts
        
        # Analyse du portefeuille
        risk_analysis = self.analyze_portfolio_risk()
        
        # Alerte VaR dépassé
        if risk_analysis['var_percentage'] > 2.0:
            alerts.append({
                'type': 'HIGH_VAR',
                'severity': 'HIGH',
                'message': f"VaR portefeuille élevé: {risk_analysis['var_percentage']:.1f}% (>2%)",
                'action': 'Réduire les positions à risque'
            })
        
        # Alerte concentration
        if risk_analysis['largest_position_weight'] > 25:
            alerts.append({
                'type': 'CONCENTRATION',
                'severity': 'MEDIUM',
                'message': f"Position trop concentrée: {risk_analysis['largest_position_weight']:.1f}% (>25%)",
                'action': 'Diversifier le portefeuille'
            })
        
        # Alerte faible diversification
        if risk_analysis['effective_positions'] < 3:
            alerts.append({
                'type': 'LOW_DIVERSIFICATION',
                'severity': 'MEDIUM',
                'message': f"Faible diversification: {risk_analysis['effective_positions']:.1f} positions effectives",
                'action': 'Ajouter des actifs non corrélés'
            })
        
        # Tests de stress
        stress_results = self.stress_test_portfolio()
        for scenario, result in stress_results.items():
            if result['loss_percentage'] > 30:
                alerts.append({
                    'type': 'STRESS_TEST',
                    'severity': 'HIGH',
                    'message': f"Scénario {scenario}: perte potentielle {result['loss_percentage']:.1f}%",
                    'action': 'Renforcer la couverture'
                })
        
        return alerts
    
    def update_portfolio(self, positions):
        """Mise à jour du portefeuille"""
        
        self.portfolio_positions = positions.copy()
        total_value = sum(positions.values())
        self.portfolio_weights = {symbol: value/total_value 
                                for symbol, value in positions.items()}
        
        print(f"📊 Portefeuille mis à jour: {len(positions)} positions, valeur ${total_value:,.2f}")
    
    def optimize_portfolio_allocation(self, expected_returns=None, max_risk=0.15):
        """Optimisation allocation portefeuille (Markowitz)"""
        
        if len(self.returns_data) < 2:
            return {"error": "Données insuffisantes pour optimisation"}
        
        symbols = list(self.returns_data.keys())
        returns_matrix = pd.DataFrame({symbol: self.returns_data[symbol] 
                                     for symbol in symbols})
        
        # Rendements moyens annualisés
        if expected_returns is None:
            expected_returns = returns_matrix.mean() * 252
        
        # Matrice de covariance annualisée
        cov_matrix = returns_matrix.cov() * 252
        
        n_assets = len(symbols)
        
        # Fonction objectif: minimiser la variance
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Contraintes
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Somme = 1
        ]
        
        # Bornes (0% à 30% par actif)
        bounds = [(0, 0.3) for _ in range(n_assets)]
        
        # Poids initiaux égaux
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimisation
        try:
            result = minimize(
                objective, 
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                optimal_risk = np.sqrt(objective(optimal_weights))
                optimal_return = np.dot(optimal_weights, expected_returns)
                
                # Allocation en dollars
                allocations = {}
                for i, symbol in enumerate(symbols):
                    allocation_value = self.available_capital * optimal_weights[i]
                    if allocation_value > 100:  # Minimum $100
                        allocations[symbol] = {
                            'weight': optimal_weights[i],
                            'value': allocation_value,
                            'percentage': optimal_weights[i] * 100
                        }
                
                return {
                    'optimal_weights': dict(zip(symbols, optimal_weights)),
                    'expected_return': optimal_return,
                    'expected_risk': optimal_risk,
                    'sharpe_ratio': optimal_return / optimal_risk if optimal_risk > 0 else 0,
                    'allocations': allocations
                }
            else:
                return {"error": "Optimisation échouée"}
                
        except Exception as e:
            return {"error": f"Erreur optimisation: {e}"}

    def create_risk_dashboard(self):
        """Création dashboard de visualisation des risques"""
        
        if not self.portfolio_positions:
            print("❌ Aucune position pour créer le dashboard")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🛡️ SLM TRADE - Dashboard de Gestion des Risques', fontsize=16, fontweight='bold')
        
        # 1. Répartition du portefeuille
        ax1 = axes[0, 0]
        symbols = list(self.portfolio_positions.keys())
        values = list(self.portfolio_positions.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(symbols)))
        
        wedges, texts, autotexts = ax1.pie(values, labels=symbols, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('Répartition du Portefeuille')
        
        # 2. VaR par position
        ax2 = axes[0, 1]
        individual_vars = []
        for symbol in symbols:
            if symbol in self.returns_data:
                returns = self.returns_data[symbol]
                var = self.calculate_var(returns, 0.95)
                position_var = self.portfolio_positions[symbol] * var
                individual_vars.append(position_var)
            else:
                individual_vars.append(0)
        
        bars = ax2.bar(symbols, individual_vars, color=colors)
        ax2.set_title('VaR 95% par Position')
        ax2.set_ylabel('VaR ($)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Matrice de corrélation
        ax3 = axes[1, 0]
        if self.correlation_matrix is not None and len(self.correlation_matrix) > 1:
            sns.heatmap(self.correlation_matrix, annot=True, cmap='RdYlBu_r', 
                       center=0, ax=ax3, square=True)
            ax3.set_title('Matrice de Corrélation')
        else:
            ax3.text(0.5, 0.5, 'Données insuffisantes\npour corrélation', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Matrice de Corrélation')
        
        # 4. Analyse de risque
        ax4 = axes[1, 1]
        risk_analysis = self.analyze_portfolio_risk()
        
        risk_metrics = [
            f"VaR Portfolio: ${risk_analysis.get('portfolio_var_95', 0):,.0f}",
            f"VaR %: {risk_analysis.get('var_percentage', 0):.1f}%",
            f"Diversification: {risk_analysis.get('diversification_benefit', 0):.1f}%",
            f"Positions effectives: {risk_analysis.get('effective_positions', 0):.1f}",
            f"Plus grosse position: {risk_analysis.get('largest_position_weight', 0):.1f}%"
        ]
        
        ax4.text(0.1, 0.8, '\n'.join(risk_metrics), transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top')
        ax4.set_title('Métriques de Risque')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def generate_risk_report(self):
        """Génération rapport complet de risque"""
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'portfolio_summary': {
                'total_value': sum(self.portfolio_positions.values()),
                'number_of_positions': len(self.portfolio_positions),
                'available_capital': self.available_capital
            }
        }
        
        # Analyse de risque
        if self.portfolio_positions:
            risk_analysis = self.analyze_portfolio_risk()
            report['risk_analysis'] = risk_analysis
            
            # Tests de stress
            stress_results = self.stress_test_portfolio()
            report['stress_tests'] = stress_results
            
            # Alertes
            alerts = self.generate_risk_alerts()
            report['alerts'] = alerts
            
            # Optimisation recommandée
            optimization = self.optimize_portfolio_allocation()
            report['optimization_recommendation'] = optimization
        
        return report


def demo_risk_management():
    """Démonstration du module de gestion des risques"""
    
    print("🚀 DÉMONSTRATION SLM TRADE - MODULE 5: GESTION DES RISQUES")
    print("=" * 60)
    
    # Initialisation
    risk_manager = SLMRiskManager()
    risk_manager.set_capital(100000)
    
    # Portfolio d'exemple
    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    print(f"\n📊 Test avec portfolio: {test_symbols}")
    
    # Récupération des données
    risk_manager.fetch_risk_data(test_symbols)
    
    # Portfolio positions d'exemple
    portfolio_positions = {
        'AAPL': 25000,
        'GOOGL': 20000,
        'MSFT': 18000,
        'TSLA': 15000,
        'NVDA': 12000
    }
    
    risk_manager.update_portfolio(portfolio_positions)
    
    # 1. Analyse de risque portefeuille
    print("\n🛡️ ANALYSE DE RISQUE PORTEFEUILLE")
    print("-" * 40)
    risk_analysis = risk_manager.analyze_portfolio_risk()
    
    print(f"Valeur portefeuille: ${risk_analysis['portfolio_value']:,.2f}")
    print(f"VaR 95% (1 jour): ${risk_analysis['portfolio_var_95']:,.2f}")
    print(f"VaR en %: {risk_analysis['var_percentage']:.2f}%")
    print(f"Bénéfice diversification: {risk_analysis['diversification_benefit']:.1f}%")
    print(f"Positions effectives: {risk_analysis['effective_positions']:.1f}")
    
    # 2. Position sizing optimal
    print("\n💰 POSITION SIZING OPTIMAL")
    print("-" * 40)
    for symbol in ['AAPL', 'TSLA']:
        pos_size = risk_manager.calculate_optimal_position_size(symbol)
        print(f"{symbol}:")
        print(f"  Taille recommandée: ${pos_size['position_value']:,.0f}")
        print(f"  Nombre d'actions: {pos_size['shares']}")
        print(f"  Poids portfolio: {pos_size['weight']*100:.1f}%")
        print(f"  Volatilité: {pos_size.get('asset_volatility', 0)*100:.1f}%")
    
    # 3. Tests de stress
    print("\n⚠️ TESTS DE STRESS")
    print("-" * 40)
    stress_results = risk_manager.stress_test_portfolio()
    
    for scenario, result in stress_results.items():
        print(f"{scenario}:")
        print(f"  Perte: ${result['loss_amount']:,.0f} ({result['loss_percentage']:.1f}%)")
        print(f"  Capital restant: ${result['remaining_capital']:,.0f}")
    
    # 4. Alertes de risque
    print("\n🚨 ALERTES DE RISQUE")
    print("-" * 40)
    alerts = risk_manager.generate_risk_alerts()
    
    if alerts:
        for alert in alerts:
            severity_icon = "🔴" if alert['severity'] == 'HIGH' else "🟡"
            print(f"{severity_icon} {alert['message']}")
            print(f"   Action: {alert['action']}")
    else:
        print("✅ Aucune alerte de risque détectée")
    
    # 5. Optimisation portefeuille
    print("\n🎯 OPTIMISATION PORTEFEUILLE")
    print("-" * 40)
    optimization = risk_manager.optimize_portfolio_allocation()
    
    if 'error' not in optimization:
        print(f"Rendement attendu: {optimization['expected_return']*100:.2f}%")
        print(f"Risque attendu: {optimization['expected_risk']*100:.2f}%")
        print(f"Ratio de Sharpe: {optimization['sharpe_ratio']:.2f}")
        
        print("\nAllocation optimale recommandée:")
        for symbol, alloc in optimization['allocations'].items():
            print(f"  {symbol}: {alloc['percentage']:.1f}% (${alloc['value']:,.0f})")
    
    # 6. Dashboard visuel
    print("\n📊 Génération du dashboard de risque...")
    try:
        risk_manager.create_risk_dashboard()
    except:
        print("Dashboard non disponible dans cet environnement")
    
    # 7. Rapport complet
    print("\n📋 GÉNÉRATION RAPPORT COMPLET")
    print("-" * 40)
    report = risk_manager.generate_risk_report()
    print(f"Rapport généré le: {report['timestamp']}")
    print(f"Nombre d'alertes: {len(report.get('alerts', []))}")
    
    print("\n🎉 Démonstration terminée!")
    return risk_manager

# Lancement de la démonstration
if __name__ == "__main__":
    demo_risk_manager = demo_risk_management()