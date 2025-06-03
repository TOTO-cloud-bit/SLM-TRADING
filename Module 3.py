// SLM TRADE - Module 3: Intégration TradingView Complète
// Graphiques professionnels avec indicateurs techniques avancés

class TradingViewManager {
    constructor() {
        this.widget = null;
        this.activeSymbol = 'BYBIT:BTCUSDT';
        this.indicators = new Map();
        this.patterns = [];
        this.alerts = [];
        this.chartData = [];
        this.timeframe = '15';
        
        this.init();
    }

    async init() {
        await this.loadTradingViewLibrary();
        this.createChart();
        this.setupIndicators();
        this.initializePatternDetection();
        console.log('📊 TradingView Module initialisé');
    }

    async loadTradingViewLibrary() {
        return new Promise((resolve) => {
            if (window.TradingView) {
                resolve();
                return;
            }

            const script = document.createElement('script');
            script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
            script.async = true;
            script.onload = resolve;
            document.head.appendChild(script);
        });
    }

    createChart() {
        const chartContainer = document.getElementById('tradingview-chart') || this.createChartContainer();
        
        // Configuration avancée du widget TradingView
        this.widget = new TradingView.widget({
            "autosize": true,
            "symbol": this.activeSymbol,
            "interval": this.timeframe,
            "timezone": "Europe/Paris",
            "theme": "dark",
            "style": "1",
            "locale": "fr",
            "toolbar_bg": "#1a1a1a",
            "enable_publishing": false,
            "hide_top_toolbar": false,
            "hide_legend": false,
            "save_image": false,
            "container_id": "tradingview-chart",
            "studies": [
                "RSI@tv-basicstudies",
                "MACD@tv-basicstudies",
                "BB@tv-basicstudies",
                "Volume@tv-basicstudies"
            ],
            "overrides": {
                "paneProperties.background": "#0d1421",
                "paneProperties.vertGridProperties.color": "#1e293b",
                "paneProperties.horzGridProperties.color": "#1e293b",
                "symbolWatermarkProperties.transparency": 90,
                "scalesProperties.textColor": "#64748b",
                "mainSeriesProperties.candleStyle.upColor": "#22c55e",
                "mainSeriesProperties.candleStyle.downColor": "#ef4444",
                "mainSeriesProperties.candleStyle.borderUpColor": "#22c55e",
                "mainSeriesProperties.candleStyle.borderDownColor": "#ef4444",
            },
            "studies_overrides": {
                "volume.volume.color.0": "#ef444480",
                "volume.volume.color.1": "#22c55e80",
                "RSI.RSI.color": "#3b82f6",
                "MACD.MACD.color": "#8b5cf6",
                "MACD.signal.color": "#f59e0b",
                "Bollinger Bands.median.color": "#64748b",
                "Bollinger Bands.upper.color": "#f59e0b",
                "Bollinger Bands.lower.color": "#f59e0b"
            }
        });

        this.widget.onChartReady(() => {
            console.log('📈 Graphique TradingView chargé');
            this.setupRealtimeData();
            this.addCustomIndicators();
        });
    }

    createChartContainer() {
        const container = document.createElement('div');
        container.id = 'tradingview-chart';
        container.style.cssText = `
            height: 600px;
            width: 100%;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-radius: 12px;
            border: 1px solid #334155;
            margin: 10px 0;
        `;
        
        // Insérer dans l'interface principale
        const dashboardContent = document.querySelector('.dashboard-content') || document.body;
        dashboardContent.appendChild(container);
        return container;
    }

    setupRealtimeData() {
        // Connexion aux données temps réel via WebSocket Bybit
        if (window.bybitAPI) {
            // Abonnement aux données de prix
            window.bybitAPI.subscribeToTicker(this.activeSymbol.split(':')[1], (data) => {
                this.updateChartData(data);
            });

            // Abonnement aux données de profondeur
            window.bybitAPI.subscribeToOrderbook(this.activeSymbol.split(':')[1], (data) => {
                this.updateOrderbookAnalysis(data);
            });
        }
    }

    addCustomIndicators() {
        // Indicateur SLM personnalisé
        this.addSLMIndicator();
        
        // Indicateur de Force du Trend
        this.addTrendStrengthIndicator();
        
        // Indicateur de Support/Résistance automatique
        this.addSupportResistanceIndicator();
        
        // Volume Profile
        this.addVolumeProfileIndicator();
    }

    addSLMIndicator() {
        const slmStudy = {
            name: "SLM Signal",
            metainfo: {
                _metainfoVersion: 51,
                id: "SLM_Signal@tv-basicstudies",
                description: "SLM Trading Signal",
                shortDescription: "SLM",
                format: {
                    type: "price",
                    precision: 4
                },
                plots: [{
                    id: "signal",
                    type: "line"
                }],
                defaults: {
                    styles: {
                        signal: {
                            linestyle: 0,
                            linewidth: 2,
                            plottype: 1,
                            trackPrice: false,
                            transparency: 0,
                            color: "#00d4ff"
                        }
                    }
                },
                styles: {
                    signal: {
                        title: "SLM Signal",
                        histogramBase: 0
                    }
                },
                inputs: []
            },
            constructor: function() {
                this.main = function(context, inputCallback) {
                    this._context = context;
                    this._input = inputCallback;
                    
                    const close = this._input(0);
                    const high = this._input(1);
                    const low = this._input(2);
                    const volume = this._input(3);
                    
                    // Calcul SLM personnalisé
                    const slmValue = this.calculateSLM(close, high, low, volume);
                    return [slmValue];
                };
                
                this.calculateSLM = function(close, high, low, volume) {
                    // Algorithme SLM propriétaire
                    const hl2 = (high + low) / 2;
                    const hlc3 = (high + low + close) / 3;
                    const volumeWeight = Math.log(volume) / 10;
                    
                    return hlc3 * (1 + volumeWeight * 0.1);
                };
            }
        };

        this.indicators.set('SLM', slmStudy);
    }

    addTrendStrengthIndicator() {
        // Force du trend basé sur ADX amélioré
        const trendStrength = {
            calculate: (highs, lows, closes, period = 14) => {
                const tr = [];
                const plusDM = [];
                const minusDM = [];
                
                for (let i = 1; i < closes.length; i++) {
                    const high = highs[i];
                    const low = lows[i];
                    const close = closes[i];
                    const prevHigh = highs[i-1];
                    const prevLow = lows[i-1];
                    const prevClose = closes[i-1];
                    
                    // True Range
                    tr.push(Math.max(
                        high - low,
                        Math.abs(high - prevClose),
                        Math.abs(low - prevClose)
                    ));
                    
                    // Directional Movement
                    const highDiff = high - prevHigh;
                    const lowDiff = prevLow - low;
                    
                    plusDM.push(highDiff > lowDiff && highDiff > 0 ? highDiff : 0);
                    minusDM.push(lowDiff > highDiff && lowDiff > 0 ? lowDiff : 0);
                }
                
                // Calcul ADX amélioré
                return this.calculateEnhancedADX(tr, plusDM, minusDM, period);
            }
        };
        
        this.indicators.set('TrendStrength', trendStrength);
    }

    addSupportResistanceIndicator() {
        const srLevels = {
            calculate: (highs, lows, closes, lookback = 20, minTouches = 3) => {
                const levels = [];
                
                for (let i = lookback; i < closes.length - lookback; i++) {
                    // Recherche des pivots
                    const isHighPivot = this.isPivotHigh(highs, i, lookback);
                    const isLowPivot = this.isPivotLow(lows, i, lookback);
                    
                    if (isHighPivot) {
                        const level = {
                            price: highs[i],
                            type: 'resistance',
                            strength: this.calculateLevelStrength(highs, lows, closes, highs[i], minTouches),
                            index: i
                        };
                        if (level.strength >= minTouches) levels.push(level);
                    }
                    
                    if (isLowPivot) {
                        const level = {
                            price: lows[i],
                            type: 'support',
                            strength: this.calculateLevelStrength(highs, lows, closes, lows[i], minTouches),
                            index: i
                        };
                        if (level.strength >= minTouches) levels.push(level);
                    }
                }
                
                return levels.sort((a, b) => b.strength - a.strength);
            }
        };
        
        this.indicators.set('SupportResistance', srLevels);
    }

    addVolumeProfileIndicator() {
        const volumeProfile = {
            calculate: (highs, lows, volumes, bins = 50) => {
                const profile = [];
                const priceRange = Math.max(...highs) - Math.min(...lows);
                const binSize = priceRange / bins;
                const minPrice = Math.min(...lows);
                
                // Initialiser les bins
                for (let i = 0; i < bins; i++) {
                    profile.push({
                        price: minPrice + (i * binSize),
                        volume: 0,
                        buyVolume: 0,
                        sellVolume: 0
                    });
                }
                
                // Distribuer les volumes
                for (let i = 0; i < highs.length; i++) {
                    const high = highs[i];
                    const low = lows[i];
                    const volume = volumes[i];
                    const avgPrice = (high + low) / 2;
                    
                    const binIndex = Math.floor((avgPrice - minPrice) / binSize);
                    if (binIndex >= 0 && binIndex < bins) {
                        profile[binIndex].volume += volume;
                        // Estimation buy/sell basée sur la position dans la bougie
                        const closePercent = (closes[i] - low) / (high - low);
                        profile[binIndex].buyVolume += volume * closePercent;
                        profile[binIndex].sellVolume += volume * (1 - closePercent);
                    }
                }
                
                return profile;
            }
        };
        
        this.indicators.set('VolumeProfile', volumeProfile);
    }

    detectPatterns(ohlcData) {
        const patterns = [];
        
        // Détection de patterns de chandeliers
        patterns.push(...this.detectCandlestickPatterns(ohlcData));
        
        // Détection de patterns techniques
        patterns.push(...this.detectTechnicalPatterns(ohlcData));
        
        // Détection de patterns harmoniques
        patterns.push(...this.detectHarmonicPatterns(ohlcData));
        
        return patterns;
    }

    detectCandlestickPatterns(data) {
        const patterns = [];
        
        for (let i = 2; i < data.length; i++) {
            const current = data[i];
            const prev = data[i-1];
            const prev2 = data[i-2];
            
            // Doji
            if (this.isDoji(current)) {
                patterns.push({
                    name: 'Doji',
                    type: 'reversal',
                    reliability: 0.6,
                    index: i,
                    signal: 'indecision'
                });
            }
            
            // Hammer / Hanging Man
            const hammerResult = this.isHammer(current, prev);
            if (hammerResult) {
                patterns.push({
                    name: hammerResult.name,
                    type: 'reversal',
                    reliability: 0.7,
                    index: i,
                    signal: hammerResult.signal
                });
            }
            
            // Engulfing Pattern
            const engulfing = this.isEngulfing(current, prev);
            if (engulfing) {
                patterns.push({
                    name: engulfing.name,
                    type: 'reversal',
                    reliability: 0.8,
                    index: i,
                    signal: engulfing.signal
                });
            }
            
            // Three White Soldiers / Three Black Crows
            const threePattern = this.isThreePattern(current, prev, prev2);
            if (threePattern) {
                patterns.push({
                    name: threePattern.name,
                    type: 'continuation',
                    reliability: 0.85,
                    index: i,
                    signal: threePattern.signal
                });
            }
        }
        
        return patterns;
    }

    detectTechnicalPatterns(data) {
        const patterns = [];
        
        // Head and Shoulders
        const headShoulders = this.detectHeadAndShoulders(data);
        patterns.push(...headShoulders);
        
        // Double Top/Bottom
        const doubles = this.detectDoubleTopBottom(data);
        patterns.push(...doubles);
        
        // Triangle Patterns
        const triangles = this.detectTriangles(data);
        patterns.push(...triangles);
        
        // Flag and Pennant
        const flags = this.detectFlagsAndPennants(data);
        patterns.push(...flags);
        
        return patterns;
    }

    detectHarmonicPatterns(data) {
        const patterns = [];
        
        // Gartley Pattern
        const gartley = this.detectGartley(data);
        patterns.push(...gartley);
        
        // Butterfly Pattern
        const butterfly = this.detectButterfly(data);
        patterns.push(...butterfly);
        
        // Bat Pattern
        const bat = this.detectBat(data);
        patterns.push(...bat);
        
        // Crab Pattern
        const crab = this.detectCrab(data);
        patterns.push(...crab);
        
        return patterns;
    }

    // Méthodes utilitaires pour la détection de patterns
    isDoji(candle) {
        const bodySize = Math.abs(candle.close - candle.open);
        const totalRange = candle.high - candle.low;
        return bodySize / totalRange < 0.1;
    }

    isHammer(candle, prevCandle) {
        const bodySize = Math.abs(candle.close - candle.open);
        const lowerShadow = Math.min(candle.open, candle.close) - candle.low;
        const upperShadow = candle.high - Math.max(candle.open, candle.close);
        const totalRange = candle.high - candle.low;
        
        if (lowerShadow > bodySize * 2 && upperShadow < bodySize * 0.5) {
            const trend = this.getTrend(prevCandle);
            if (trend === 'down') {
                return { name: 'Hammer', signal: 'bullish' };
            } else if (trend === 'up') {
                return { name: 'Hanging Man', signal: 'bearish' };
            }
        }
        return null;
    }

    isEngulfing(current, prev) {
        const currentBull = current.close > current.open;
        const prevBull = prev.close > prev.open;
        
        if (currentBull && !prevBull) {
            // Bullish Engulfing
            if (current.open < prev.close && current.close > prev.open) {
                return { name: 'Bullish Engulfing', signal: 'bullish' };
            }
        } else if (!currentBull && prevBull) {
            // Bearish Engulfing
            if (current.open > prev.close && current.close < prev.open) {
                return { name: 'Bearish Engulfing', signal: 'bearish' };
            }
        }
        return null;
    }

    createPatternAlert(pattern) {
        const alert = {
            id: Date.now(),
            timestamp: new Date(),
            pattern: pattern.name,
            signal: pattern.signal,
            reliability: pattern.reliability,
            price: this.getCurrentPrice(),
            message: `Pattern ${pattern.name} détecté - Signal ${pattern.signal}`,
            type: pattern.reliability > 0.7 ? 'high' : 'medium'
        };
        
        this.alerts.push(alert);
        this.showPatternNotification(alert);
        
        // Callback vers l'interface principale
        if (window.updatePatternDetection) {
            window.updatePatternDetection(pattern, alert);
        }
        
        return alert;
    }

    showPatternNotification(alert) {
        // Créer une notification toast
        const notification = document.createElement('div');
        notification.className = `pattern-notification ${alert.type}`;
        notification.innerHTML = `
            <div class="notification-icon">📊</div>
            <div class="notification-content">
                <div class="notification-title">Pattern Détecté</div>
                <div class="notification-message">${alert.message}</div>
                <div class="notification-time">${alert.timestamp.toLocaleTimeString()}</div>
            </div>
        `;
        
        // Styles
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border: 1px solid #00d4ff;
            border-radius: 8px;
            padding: 15px;
            color: #ffffff;
            box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
            z-index: 10000;
            display: flex;
            align-items: center;
            gap: 10px;
            animation: slideInRight 0.3s ease-out;
            max-width: 300px;
        `;
        
        document.body.appendChild(notification);
        
        // Suppression automatique après 5 secondes
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }

    // Méthodes d'analyse technique avancée
    calculateRSI(closes, period = 14) {
        const gains = [];
        const losses = [];
        
        for (let i = 1; i < closes.length; i++) {
            const change = closes[i] - closes[i-1];
            gains.push(change > 0 ? change : 0);
            losses.push(change < 0 ? Math.abs(change) : 0);
        }
        
        const avgGain = gains.slice(-period).reduce((a, b) => a + b) / period;
        const avgLoss = losses.slice(-period).reduce((a, b) => a + b) / period;
        
        if (avgLoss === 0) return 100;
        
        const rs = avgGain / avgLoss;
        return 100 - (100 / (1 + rs));
    }

    calculateMACD(closes, fast = 12, slow = 26, signal = 9) {
        const emaFast = this.calculateEMA(closes, fast);
        const emaSlow = this.calculateEMA(closes, slow);
        
        const macdLine = emaFast.map((fast, i) => fast - emaSlow[i]);
        const signalLine = this.calculateEMA(macdLine, signal);
        const histogram = macdLine.map((macd, i) => macd - signalLine[i]);
        
        return {
            macd: macdLine,
            signal: signalLine,
            histogram: histogram
        };
    }

    calculateEMA(data, period) {
        const ema = [];
        const multiplier = 2 / (period + 1);
        
        ema[0] = data[0];
        
        for (let i = 1; i < data.length; i++) {
            ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier));
        }
        
        return ema;
    }

    calculateBollingerBands(closes, period = 20, stdDev = 2) {
        const sma = this.calculateSMA(closes, period);
        const bands = {
            upper: [],
            middle: sma,
            lower: []
        };
        
        for (let i = period - 1; i < closes.length; i++) {
            const slice = closes.slice(i - period + 1, i + 1);
            const mean = sma[i];
            const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period;
            const standardDeviation = Math.sqrt(variance);
            
            bands.upper[i] = mean + (standardDeviation * stdDev);
            bands.lower[i] = mean - (standardDeviation * stdDev);
        }
        
        return bands;
    }

    calculateSMA(data, period) {
        const sma = [];
        
        for (let i = period - 1; i < data.length; i++) {
            const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b);
            sma[i] = sum / period;
        }
        
        return sma;
    }

    // Interface de contrôle
    changeSymbol(symbol) {
        this.activeSymbol = symbol;
        if (this.widget) {
            this.widget.setSymbol(symbol, () => {
                console.log(`📊 Symbole changé vers: ${symbol}`);
                this.setupRealtimeData();
            });
        }
    }

    changeTimeframe(timeframe) {
        this.timeframe = timeframe;
        if (this.widget) {
            this.widget.chart().setResolution(timeframe, () => {
                console.log(`⏱️ Timeframe changé vers: ${timeframe}`);
            });
        }
    }

    addIndicator(indicatorName, params = {}) {
        if (this.widget && this.widget.chart) {
            this.widget.chart().createStudy(indicatorName, false, false, params);
            console.log(`📈 Indicateur ajouté: ${indicatorName}`);
        }
    }

    removeIndicator(indicatorId) {
        if (this.widget && this.widget.chart) {
            this.widget.chart().removeEntity(indicatorId);
            console.log(`❌ Indicateur supprimé: ${indicatorId}`);
        }
    }

    takeScreenshot() {
        if (this.widget && this.widget.chart) {
            this.widget.chart().takeScreenshot();
            console.log('📸 Capture d\'écran du graphique prise');
        }
    }

    exportData() {
        const exportData = {
            symbol: this.activeSymbol,
            timeframe: this.timeframe,
            indicators: Array.from(this.indicators.keys()),
            patterns: this.patterns,
            alerts: this.alerts,
            timestamp: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `slm-trade-analysis-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        console.log('💾 Données d\'analyse exportées');
    }

    // Méthodes utilitaires
    getCurrentPrice() {
        return this.chartData.length > 0 ? this.chartData[this.chartData.length - 1].close : 0;
    }

    getTrend(candle, period = 10) {
        // Logique simplifiée pour détecter la tendance
        if (candle.close > candle.open) return 'up';
        if (candle.close < candle.open) return 'down';
        return 'neutral';
    }

    isPivotHigh(highs, index, lookback) {
        const currentHigh = highs[index];
        for (let i = index - lookback; i <= index + lookback; i++) {
            if (i !== index && i >= 0 && i < highs.length) {
                if (highs[i] >= currentHigh) return false;
            }
        }
        return true;
    }

    isPivotLow(lows, index, lookback) {
        const currentLow = lows[index];
        for (let i = index - lookback; i <= index + lookback; i++) {
            if (i !== index && i >= 0 && i < lows.length) {
                if (lows[i] <= currentLow) return false;
            }
        }
        return true;
    }

    calculateLevelStrength(highs, lows, closes, level, tolerance = 0.001) {
        let touches = 0;
        const toleranceAmount = level * tolerance;
        
        for (let i = 0; i < closes.length; i++) {
            if (Math.abs(highs[i] - level) <= toleranceAmount ||
                Math.abs(lows[i] - level) <= toleranceAmount ||
                Math.abs(closes[i] - level) <= toleranceAmount) {
                touches++;
            }
        }
        
        return touches;
    }

    // Interface de gestion des alertes
    createAlert(condition, message, type = 'info') {
        const alert = {
            id: Date.now(),
            condition: condition,
            message: message,
            type: type,
            active: true,
            triggered: false,
            createdAt: new Date()
        };
        
        this.alerts.push(alert);
        return alert;
    }

    checkAlerts() {
        const currentPrice = this.getCurrentPrice();
        
        this.alerts.filter(alert => alert.active && !alert.triggered).forEach(alert => {
            if (this.evaluateAlertCondition(alert.condition, currentPrice)) {
                alert.triggered = true;
                this.triggerAlert(alert);
            }
        });
    }

    evaluateAlertCondition(condition, currentPrice) {
        // Évaluation simple des conditions d'alerte
        try {
            return eval(condition.replace('PRICE', currentPrice));
        } catch (error) {
            console.error('Erreur dans la condition d\'alerte:', error);
            return false;
        }
    }

    triggerAlert(alert) {
        console.log(`🚨 Alerte déclenchée: ${alert.message}`);
        this.showPatternNotification({
            type: alert.type,
            message: alert.message,
            timestamp: new Date()
        });
        
        // Callback vers l'interface principale
        if (window.onAlertTriggered) {
            window.onAlertTriggered(alert);
        }
    }
}

// Initialisation automatique
let tradingViewManager;

document.addEventListener('DOMContentLoaded', () => {
    tradingViewManager = new TradingViewManager();
});

// Fonctions d'interface globales
window.changeTradingSymbol = (symbol) => {
    if (tradingViewManager) {
        tradingViewManager.changeSymbol(symbol);
    }
};

window.changeTradingTimeframe = (timeframe) => {
    if (tradingViewManager) {
        tradingViewManager.changeTimeframe(timeframe);
    }
};

window.addTradingIndicator = (indicator, params) => {
    if (tradingViewManager) {
        tradingViewManager.addIndicator(indicator, params);
    }
};

window.exportTradingData = () => {
    if (tradingViewManager) {
        tradingViewManager.exportData();
    }
};

window.createTradingAlert = (condition, message, type) => {
    if (tradingViewManager) {
        return tradingViewManager.createAlert(condition, message, type);
    }
};

// Styles CSS pour les notifications
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .pattern-notification {
        font-family: 'Inter', sans-serif;
    }
    
    .pattern-notification.high {
        border-color: #22c55e;
        box-shadow: 0 10px 30px rgba(34, 197, 94, 0.3);
    }
    
    .pattern-notification.medium {
        border-color: #f59e0b;
        box-shadow: 0 10px 30px rgba(245, 158, 11, 0.3);
    }
    
    .notification-icon {
        font-size: 24px;
        flex-shrink: 0;
    }
    
    .notification-content {
        flex-grow: 1;
    }
    
    .notification-title {
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 4px;
    }
    
    .notification-message {
        font-size: 12px;
        opacity: 0.8;
        margin-bottom: 2px;
    }
    
    .notification-time {
        font-size: 10px;
        opacity: 0.6;
    }
`;

document.head.appendChild(notificationStyles);

// Extension pour l'analyse de sentiment et flux d'ordres
class MarketSentimentAnalyzer {
    constructor() {
        this.orderFlow = [];
        this.sentimentScore = 0;
        this.volumeAnalysis = {
            buyPressure: 0,
            sellPressure: 0,
            netFlow: 0
        };
        this.init();
    }

    init() {
        this.startOrderFlowAnalysis();
        this.initializeSentimentTracking();
        console.log('📊 Analyseur de sentiment de marché initialisé');
    }

    startOrderFlowAnalysis() {
        // Analyse du flux d'ordres en temps réel
        if (window.bybitAPI) {
            window.bybitAPI.subscribeToTrades('BTCUSDT', (trade) => {
                this.analyzeOrderFlow(trade);
            });
        }
    }

    analyzeOrderFlow(trade) {
        const orderFlowData = {
            timestamp: trade.timestamp,
            price: parseFloat(trade.price),
            size: parseFloat(trade.size),
            side: trade.side, // 'Buy' or 'Sell'
            isBlockTrade: parseFloat(trade.size) > this.getAverageTradeSize() * 5
        };

        this.orderFlow.unshift(orderFlowData);
        
        // Garder seulement les 1000 derniers trades
        if (this.orderFlow.length > 1000) {
            this.orderFlow = this.orderFlow.slice(0, 1000);
        }

        this.updateVolumeAnalysis();
        this.calculateSentimentScore();
        this.detectLargeOrders(orderFlowData);
    }

    updateVolumeAnalysis() {
        const recentTrades = this.orderFlow.slice(0, 100); // 100 derniers trades
        
        let buyVolume = 0;
        let sellVolume = 0;
        
        recentTrades.forEach(trade => {
            if (trade.side === 'Buy') {
                buyVolume += trade.size;
            } else {
                sellVolume += trade.size;
            }
        });
        
        this.volumeAnalysis = {
            buyPressure: buyVolume,
            sellPressure: sellVolume,
            netFlow: buyVolume - sellVolume,
            dominance: buyVolume > sellVolume ? 'buyers' : 'sellers',
            ratio: sellVolume > 0 ? (buyVolume / sellVolume).toFixed(2) : 'N/A'
        };
    }

    calculateSentimentScore() {
        if (this.orderFlow.length < 50) return;
        
        const recentTrades = this.orderFlow.slice(0, 100);
        let score = 0;
        let weightedScore = 0;
        let totalWeight = 0;
        
        recentTrades.forEach((trade, index) => {
            const weight = Math.exp(-index * 0.05); // Poids décroissant
            const tradeScore = trade.side === 'Buy' ? 1 : -1;
            const sizeMultiplier = Math.log(trade.size + 1);
            
            weightedScore += tradeScore * weight * sizeMultiplier;
            totalWeight += weight * sizeMultiplier;
        });
        
        this.sentimentScore = totalWeight > 0 ? (weightedScore / totalWeight) * 100 : 0;
        
        // Callback vers l'interface
        if (window.updateSentimentDisplay) {
            window.updateSentimentDisplay(this.sentimentScore, this.volumeAnalysis);
        }
    }

    detectLargeOrders(trade) {
        if (trade.isBlockTrade) {
            const alert = {
                type: 'large_order',
                timestamp: new Date(),
                message: `Ordre important détecté: ${trade.side} ${trade.size} à ${trade.price}`,
                trade: trade
            };
            
            this.showLargeOrderAlert(alert);
            
            // Log pour analyse
            console.log('🐋 Ordre important détecté:', trade);
        }
    }

    showLargeOrderAlert(alert) {
        const notification = document.createElement('div');
        notification.className = 'large-order-notification';
        notification.innerHTML = `
            <div class="notification-icon">🐋</div>
            <div class="notification-content">
                <div class="notification-title">Ordre Important</div>
                <div class="notification-message">${alert.message}</div>
                <div class="notification-time">${alert.timestamp.toLocaleTimeString()}</div>
            </div>
        `;
        
        notification.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
            border: 1px solid #8b5cf6;
            border-radius: 8px;
            padding: 15px;
            color: #ffffff;
            box-shadow: 0 10px 30px rgba(139, 92, 246, 0.4);
            z-index: 10000;
            display: flex;
            align-items: center;
            gap: 10px;
            animation: slideInRight 0.3s ease-out;
            max-width: 300px;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => notification.remove(), 300);
        }, 7000);
    }

    getAverageTradeSize() {
        if (this.orderFlow.length === 0) return 1;
        
        const total = this.orderFlow.reduce((sum, trade) => sum + trade.size, 0);
        return total / this.orderFlow.length;
    }

    getSentimentSummary() {
        return {
            score: this.sentimentScore,
            interpretation: this.interpretSentiment(this.sentimentScore),
            volumeAnalysis: this.volumeAnalysis,
            recentActivity: this.getRecentActivity(),
            marketPressure: this.getMarketPressure()
        };
    }

    interpretSentiment(score) {
        if (score > 50) return 'Très Haussier';
        if (score > 20) return 'Haussier';
        if (score > -20) return 'Neutre';
        if (score > -50) return 'Baissier';
        return 'Très Baissier';
    }

    getRecentActivity() {
        const recent = this.orderFlow.slice(0, 20);
        return {
            averageSize: recent.reduce((sum, t) => sum + t.size, 0) / recent.length,
            priceMovement: recent.length > 1 ? recent[0].price - recent[recent.length - 1].price : 0,
            frequency: recent.length
        };
    }

    getMarketPressure() {
        const ratio = this.volumeAnalysis.ratio;
        if (ratio === 'N/A') return 'Indéterminé';
        
        const r = parseFloat(ratio);
        if (r > 2) return 'Forte pression acheteuse';
        if (r > 1.5) return 'Pression acheteuse modérée';
        if (r > 0.67) return 'Équilibré';
        if (r > 0.5) return 'Pression vendeuse modérée';
        return 'Forte pression vendeuse';
    }
}

// Gestionnaire d'alertes avancées
class AdvancedAlertManager {
    constructor() {
        this.alerts = new Map();
        this.alertHistory = [];
        this.alertTypes = {
            PRICE: 'price',
            VOLUME: 'volume',
            PATTERN: 'pattern',
            INDICATOR: 'indicator',
            SENTIMENT: 'sentiment',
            ORDER_FLOW: 'order_flow'
        };
        this.init();
    }

    init() {
        this.setupDefaultAlerts();
        this.startAlertMonitoring();
        console.log('🚨 Gestionnaire d\'alertes avancées initialisé');
    }

    setupDefaultAlerts() {
        // Alertes de prix par défaut
        this.createAlert({
            name: 'Support cassé',
            type: this.alertTypes.PRICE,
            condition: 'PRICE < SUPPORT_LEVEL',
            message: 'Le prix a cassé le niveau de support',
            priority: 'high',
            autoTrade: false
        });

        this.createAlert({
            name: 'Résistance franchie',
            type: this.alertTypes.PRICE,
            condition: 'PRICE > RESISTANCE_LEVEL',
            message: 'Le prix a franchi le niveau de résistance',
            priority: 'high',
            autoTrade: false
        });

        // Alertes de volume
        this.createAlert({
            name: 'Volume anormal',
            type: this.alertTypes.VOLUME,
            condition: 'VOLUME > AVERAGE_VOLUME * 3',
            message: 'Volume anormalement élevé détecté',
            priority: 'medium',
            autoTrade: false
        });
    }

    createAlert(alertConfig) {
        const alert = {
            id: Date.now() + Math.random(),
            ...alertConfig,
            created: new Date(),
            triggered: false,
            triggerCount: 0,
            lastTriggered: null,
            active: true
        };

        this.alerts.set(alert.id, alert);
        return alert;
    }

    startAlertMonitoring() {
        setInterval(() => {
            this.checkAllAlerts();
        }, 1000); // Vérification chaque seconde
    }

    checkAllAlerts() {
        const currentData = this.getCurrentMarketData();
        
        this.alerts.forEach(alert => {
            if (alert.active && this.shouldCheckAlert(alert)) {
                if (this.evaluateAlertCondition(alert, currentData)) {
                    this.triggerAlert(alert, currentData);
                }
            }
        });
    }

    shouldCheckAlert(alert) {
        // Éviter les alertes trop fréquentes
        if (alert.lastTriggered) {
            const timeSinceLastTrigger = Date.now() - alert.lastTriggered.getTime();
            const cooldownPeriod = this.getCooldownPeriod(alert.priority);
            return timeSinceLastTrigger > cooldownPeriod;
        }
        return true;
    }

    getCooldownPeriod(priority) {
        switch (priority) {
            case 'high': return 30000; // 30 secondes
            case 'medium': return 60000; // 1 minute
            case 'low': return 300000; // 5 minutes
            default: return 60000;
        }
    }

    evaluateAlertCondition(alert, marketData) {
        try {
            let condition = alert.condition;
            
            // Remplacer les variables par les valeurs réelles
            condition = condition.replace(/PRICE/g, marketData.price);
            condition = condition.replace(/VOLUME/g, marketData.volume);
            condition = condition.replace(/RSI/g, marketData.rsi || 50);
            condition = condition.replace(/MACD/g, marketData.macd || 0);
            condition = condition.replace(/SUPPORT_LEVEL/g, marketData.supportLevel || 0);
            condition = condition.replace(/RESISTANCE_LEVEL/g, marketData.resistanceLevel || 999999);
            condition = condition.replace(/AVERAGE_VOLUME/g, marketData.averageVolume || 1000);
            
            return eval(condition);
        } catch (error) {
            console.error('Erreur évaluation alerte:', error);
            return false;
        }
    }

    triggerAlert(alert, marketData) {
        alert.triggered = true;
        alert.triggerCount++;
        alert.lastTriggered = new Date();

        const alertEvent = {
            alert: alert,
            marketData: marketData,
            timestamp: new Date()
        };

        this.alertHistory.unshift(alertEvent);
        
        // Garder seulement les 100 dernières alertes
        if (this.alertHistory.length > 100) {
            this.alertHistory = this.alertHistory.slice(0, 100);
        }

        this.showAlert(alertEvent);
        this.logAlert(alertEvent);

        // Auto-trading si activé
        if (alert.autoTrade && window.executeAutoTrade) {
            window.executeAutoTrade(alert, marketData);
        }

        // Callback vers l'interface
        if (window.onAdvancedAlert) {
            window.onAdvancedAlert(alertEvent);
        }
    }

    showAlert(alertEvent) {
        const { alert, marketData } = alertEvent;
        
        const notification = document.createElement('div');
        notification.className = `advanced-alert ${alert.priority}`;
        notification.innerHTML = `
            <div class="alert-icon">${this.getAlertIcon(alert.type)}</div>
            <div class="alert-content">
                <div class="alert-title">${alert.name}</div>
                <div class="alert-message">${alert.message}</div>
                <div class="alert-details">Prix: ${marketData.price} | Vol: ${marketData.volume}</div>
                <div class="alert-time">${alertEvent.timestamp.toLocaleTimeString()}</div>
            </div>
            <div class="alert-actions">
                <button onclick="window.dismissAlert('${alert.id}')" class="dismiss-btn">×</button>
            </div>
        `;
        
        const colors = {
            high: '#ef4444',
            medium: '#f59e0b',
            low: '#10b981'
        };
        
        notification.style.cssText = `
            position: fixed;
            top: ${20 + (this.getActiveNotifications() * 90)}px;
            right: 20px;
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border: 2px solid ${colors[alert.priority]};
            border-radius: 12px;
            padding: 16px;
            color: #ffffff;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
            z-index: 10001;
            display: flex;
            align-items: flex-start;
            gap: 12px;
            animation: slideInRight 0.4s ease-out;
            max-width: 350px;
            min-width: 300px;
        `;
        
        document.body.appendChild(notification);
        
        // Auto-suppression basée sur la priorité
        const autoRemoveTime = alert.priority === 'high' ? 10000 : 
                              alert.priority === 'medium' ? 7000 : 5000;
        
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.animation = 'slideOutRight 0.4s ease-in';
                setTimeout(() => notification.remove(), 400);
            }
        }, autoRemoveTime);
    }

    getAlertIcon(type) {
        const icons = {
            [this.alertTypes.PRICE]: '💰',
            [this.alertTypes.VOLUME]: '📊',
            [this.alertTypes.PATTERN]: '📈',
            [this.alertTypes.INDICATOR]: '⚡',
            [this.alertTypes.SENTIMENT]: '🎭',
            [this.alertTypes.ORDER_FLOW]: '🌊'
        };
        return icons[type] || '🚨';
    }

    getActiveNotifications() {
        return document.querySelectorAll('.advanced-alert').length;
    }

    getCurrentMarketData() {
        // Récupération des données de marché actuelles
        return {
            price: window.tradingViewManager ? window.tradingViewManager.getCurrentPrice() : 50000,
            volume: Math.random() * 1000 + 500, // Simulé pour l'exemple
            rsi: Math.random() * 100,
            macd: (Math.random() - 0.5) * 2,
            supportLevel: 48000,
            resistanceLevel: 52000,
            averageVolume: 750,
            timestamp: new Date()
        };
    }

    logAlert(alertEvent) {
        console.log(`🚨 ALERTE [${alertEvent.alert.priority.toUpperCase()}]: ${alertEvent.alert.name}`, {
            message: alertEvent.alert.message,
            price: alertEvent.marketData.price,
            time: alertEvent.timestamp.toLocaleString()
        });
    }

    // Interface de gestion des alertes
    getAlertById(id) {
        return this.alerts.get(id);
    }

    updateAlert(id, updates) {
        const alert = this.alerts.get(id);
        if (alert) {
            Object.assign(alert, updates);
            return alert;
        }
        return null;
    }

    deleteAlert(id) {
        return this.alerts.delete(id);
    }

    toggleAlert(id) {
        const alert = this.alerts.get(id);
        if (alert) {
            alert.active = !alert.active;
            return alert;
        }
        return null;
    }

    getAlertHistory(limit = 50) {
        return this.alertHistory.slice(0, limit);
    }

    getAlertStatistics() {
        const total = this.alerts.size;
        const active = Array.from(this.alerts.values()).filter(a => a.active).length;
        const triggered = Array.from(this.alerts.values()).filter(a => a.triggered).length;
        
        return {
            total,
            active,
            inactive: total - active,
            triggered,
            triggerRate: total > 0 ? (triggered / total * 100).toFixed(1) + '%' : '0%',
            recentTriggers: this.alertHistory.slice(0, 10)
        };
    }
}

// Initialisation des modules étendus
let marketSentimentAnalyzer;
let advancedAlertManager;

document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        marketSentimentAnalyzer = new MarketSentimentAnalyzer();
        advancedAlertManager = new AdvancedAlertManager();
    }, 2000);
});

// Fonctions globales pour l'interface
window.dismissAlert = (alertId) => {
    const notifications = document.querySelectorAll('.advanced-alert');
    notifications.forEach(notification => {
        if (notification.innerHTML.includes(alertId)) {
            notification.remove();
        }
    });
};

window.getSentimentAnalysis = () => {
    return marketSentimentAnalyzer ? marketSentimentAnalyzer.getSentimentSummary() : null;
};

window.createCustomAlert = (config) => {
    return advancedAlertManager ? advancedAlertManager.createAlert(config) : null;
};

window.getAlertStatistics = () => {
    return advancedAlertManager ? advancedAlertManager.getAlertStatistics() : null;
};