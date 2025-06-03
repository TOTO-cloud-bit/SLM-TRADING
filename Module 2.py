// SLM TRADE - Module 2: Connexions API Bybit Réelles
// Intégration complète avec l'API Bybit (REST + WebSocket)

class BybitAPI {
    constructor() {
        this.apiKey = '';
        this.apiSecret = '';
        this.baseURL = 'https://api.bybit.com';
        this.testnetURL = 'https://api-testnet.bybit.com';
        this.isTestnet = true; // Basculer en false pour le trading réel
        this.ws = null;
        this.isConnected = false;
        this.subscriptions = new Set();
        this.orderBook = new Map();
        this.klineData = new Map();
        this.positions = new Map();
        this.orders = new Map();
        this.callbacks = new Map();
    }

    // Configuration API
    setCredentials(apiKey, apiSecret, isTestnet = true) {
        this.apiKey = apiKey;
        this.apiSecret = apiSecret;
        this.isTestnet = isTestnet;
        this.log('API credentials configured', 'info');
    }

    // Génération de signature pour l'authentification
    generateSignature(timestamp, queryString) {
        const message = timestamp + this.apiKey + queryString;
        return CryptoJS.HmacSHA256(message, this.apiSecret).toString();
    }

    // Requête REST API authentifiée
    async makeRequest(endpoint, method = 'GET', params = {}) {
        const timestamp = Date.now().toString();
        const baseUrl = this.isTestnet ? this.testnetURL : this.baseURL;
        
        let queryString = '';
        if (method === 'GET' && Object.keys(params).length > 0) {
            queryString = new URLSearchParams(params).toString();
        }

        const signature = this.generateSignature(timestamp, queryString);
        
        const headers = {
            'X-BAPI-API-KEY': this.apiKey,
            'X-BAPI-SIGN': signature,
            'X-BAPI-TIMESTAMP': timestamp,
            'Content-Type': 'application/json'
        };

        const url = `${baseUrl}${endpoint}${queryString ? '?' + queryString : ''}`;
        
        const options = {
            method,
            headers,
        };

        if (method === 'POST' && Object.keys(params).length > 0) {
            options.body = JSON.stringify(params);
        }

        try {
            const response = await fetch(url, options);
            const data = await response.json();
            
            if (data.retCode !== 0) {
                throw new Error(`API Error: ${data.retMsg}`);
            }
            
            return data.result;
        } catch (error) {
            this.log(`API Request Error: ${error.message}`, 'error');
            throw error;
        }
    }

    // Connexion WebSocket
    async connectWebSocket() {
        const wsUrl = this.isTestnet 
            ? 'wss://stream-testnet.bybit.com/v5/public/linear'
            : 'wss://stream.bybit.com/v5/public/linear';

        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                this.isConnected = true;
                this.log('WebSocket connected successfully', 'success');
                this.updateConnectionStatus(true);
                
                // Authentification WebSocket si nécessaire
                if (this.apiKey && this.apiSecret) {
                    this.authenticateWebSocket();
                }
            };

            this.ws.onmessage = (event) => {
                this.handleWebSocketMessage(JSON.parse(event.data));
            };

            this.ws.onclose = () => {
                this.isConnected = false;
                this.log('WebSocket connection closed', 'warning');
                this.updateConnectionStatus(false);
                this.reconnectWebSocket();
            };

            this.ws.onerror = (error) => {
                this.log(`WebSocket error: ${error.message}`, 'error');
            };

        } catch (error) {
            this.log(`WebSocket connection failed: ${error.message}`, 'error');
        }
    }

    // Authentification WebSocket
    authenticateWebSocket() {
        const timestamp = Date.now();
        const signature = CryptoJS.HmacSHA256(`GET/realtime${timestamp}`, this.apiSecret).toString();
        
        const authMessage = {
            op: 'auth',
            args: [this.apiKey, timestamp, signature]
        };
        
        this.ws.send(JSON.stringify(authMessage));
    }

    // Gestion des messages WebSocket
    handleWebSocketMessage(data) {
        if (data.topic) {
            const topic = data.topic;
            
            if (topic.includes('orderbook')) {
                this.updateOrderBook(data);
            } else if (topic.includes('kline')) {
                this.updateKlineData(data);
            } else if (topic.includes('position')) {
                this.updatePositions(data);
            } else if (topic.includes('order')) {
                this.updateOrders(data);
            }
            
            // Notifier les callbacks
            if (this.callbacks.has(topic)) {
                this.callbacks.get(topic)(data);
            }
        }
    }

    // Souscription à un topic WebSocket
    subscribe(topic, callback = null) {
        if (!this.isConnected) {
            this.log('WebSocket not connected', 'error');
            return;
        }

        const subscribeMessage = {
            op: 'subscribe',
            args: [topic]
        };

        this.ws.send(JSON.stringify(subscribeMessage));
        this.subscriptions.add(topic);
        
        if (callback) {
            this.callbacks.set(topic, callback);
        }
        
        this.log(`Subscribed to ${topic}`, 'info');
    }

    // Désabonnement d'un topic
    unsubscribe(topic) {
        if (!this.isConnected) return;

        const unsubscribeMessage = {
            op: 'unsubscribe',
            args: [topic]
        };

        this.ws.send(JSON.stringify(unsubscribeMessage));
        this.subscriptions.delete(topic);
        this.callbacks.delete(topic);
        
        this.log(`Unsubscribed from ${topic}`, 'info');
    }

    // Reconnexion automatique WebSocket
    reconnectWebSocket() {
        setTimeout(() => {
            if (!this.isConnected) {
                this.log('Attempting to reconnect WebSocket...', 'info');
                this.connectWebSocket();
            }
        }, 5000);
    }

    // === FONCTIONS DE TRADING ===

    // Récupérer les informations du compte
    async getAccountBalance() {
        try {
            const result = await this.makeRequest('/v5/account/wallet-balance', 'GET', {
                accountType: 'UNIFIED'
            });
            this.log('Account balance retrieved', 'success');
            return result;
        } catch (error) {
            this.log(`Failed to get account balance: ${error.message}`, 'error');
            throw error;
        }
    }

    // Récupérer les positions
    async getPositions(symbol = '') {
        try {
            const params = {
                category: 'linear',
                settleCoin: 'USDT'
            };
            
            if (symbol) {
                params.symbol = symbol;
            }

            const result = await this.makeRequest('/v5/position/list', 'GET', params);
            this.log(`Positions retrieved${symbol ? ' for ' + symbol : ''}`, 'success');
            return result;
        } catch (error) {
            this.log(`Failed to get positions: ${error.message}`, 'error');
            throw error;
        }
    }

    // Passer un ordre
    async placeOrder(symbol, side, orderType, qty, price = null, timeInForce = 'GTC', options = {}) {
        try {
            const params = {
                category: 'linear',
                symbol: symbol,
                side: side, // 'Buy' ou 'Sell'
                orderType: orderType, // 'Market' ou 'Limit'
                qty: qty.toString(),
                timeInForce: timeInForce,
                ...options
            };

            if (orderType === 'Limit' && price) {
                params.price = price.toString();
            }

            const result = await this.makeRequest('/v5/order/create', 'POST', params);
            this.log(`Order placed: ${side} ${qty} ${symbol} at ${price || 'market'}`, 'success');
            return result;
        } catch (error) {
            this.log(`Failed to place order: ${error.message}`, 'error');
            throw error;
        }
    }

    // Annuler un ordre
    async cancelOrder(symbol, orderId) {
        try {
            const params = {
                category: 'linear',
                symbol: symbol,
                orderId: orderId
            };

            const result = await this.makeRequest('/v5/order/cancel', 'POST', params);
            this.log(`Order cancelled: ${orderId}`, 'success');
            return result;
        } catch (error) {
            this.log(`Failed to cancel order: ${error.message}`, 'error');
            throw error;
        }
    }

    // Fermer une position
    async closePosition(symbol, side) {
        try {
            const positions = await this.getPositions(symbol);
            const position = positions.list.find(p => p.symbol === symbol && p.side === side);
            
            if (!position || parseFloat(position.size) === 0) {
                throw new Error('No position to close');
            }

            const closeSide = side === 'Buy' ? 'Sell' : 'Buy';
            const result = await this.placeOrder(symbol, closeSide, 'Market', Math.abs(parseFloat(position.size)));
            
            this.log(`Position closed: ${side} ${symbol}`, 'success');
            return result;
        } catch (error) {
            this.log(`Failed to close position: ${error.message}`, 'error');
            throw error;
        }
    }

    // Récupérer l'historique des ordres
    async getOrderHistory(symbol = '', limit = 50) {
        try {
            const params = {
                category: 'linear',
                limit: limit
            };
            
            if (symbol) {
                params.symbol = symbol;
            }

            const result = await this.makeRequest('/v5/order/history', 'GET', params);
            this.log('Order history retrieved', 'success');
            return result;
        } catch (error) {
            this.log(`Failed to get order history: ${error.message}`, 'error');
            throw error;
        }
    }

    // Récupérer les données de prix
    async getTicker(symbol) {
        try {
            const params = {
                category: 'linear',
                symbol: symbol
            };

            const result = await this.makeRequest('/v5/market/tickers', 'GET', params);
            return result.list[0];
        } catch (error) {
            this.log(`Failed to get ticker for ${symbol}: ${error.message}`, 'error');
            throw error;
        }
    }

    // Récupérer les données kline/chandelier
    async getKlineData(symbol, interval = '1', limit = 200) {
        try {
            const params = {
                category: 'linear',
                symbol: symbol,
                interval: interval,
                limit: limit
            };

            const result = await this.makeRequest('/v5/market/kline', 'GET', params);
            this.log(`Kline data retrieved for ${symbol}`, 'success');
            return result;
        } catch (error) {
            this.log(`Failed to get kline data: ${error.message}`, 'error');
            throw error;
        }
    }

    // === FONCTIONS DE MISE À JOUR DES DONNÉES ===

    updateOrderBook(data) {
        const symbol = data.data.s;
        this.orderBook.set(symbol, data.data);
        
        // Mettre à jour l'interface si elle existe
        if (typeof updateOrderBookUI === 'function') {
            updateOrderBookUI(symbol, data.data);
        }
    }

    updateKlineData(data) {
        const symbol = data.data.symbol;
        if (!this.klineData.has(symbol)) {
            this.klineData.set(symbol, []);
        }
        
        const klines = this.klineData.get(symbol);
        klines.push(data.data);
        
        // Garder seulement les 1000 dernières bougies
        if (klines.length > 1000) {
            klines.shift();
        }
        
        // Mettre à jour l'interface si elle existe
        if (typeof updateChartUI === 'function') {
            updateChartUI(symbol, data.data);
        }
    }

    updatePositions(data) {
        data.data.forEach(position => {
            this.positions.set(position.symbol, position);
        });
        
        // Mettre à jour l'interface si elle existe
        if (typeof updatePositionsUI === 'function') {
            updatePositionsUI(data.data);
        }
    }

    updateOrders(data) {
        data.data.forEach(order => {
            this.orders.set(order.orderId, order);
        });
        
        // Mettre à jour l'interface si elle existe
        if (typeof updateOrdersUI === 'function') {
            updateOrdersUI(data.data);
        }
    }

    // === FONCTIONS UTILITAIRES ===

    // Test de connexion API
    async testConnection() {
        try {
            await this.makeRequest('/v5/market/time', 'GET');
            this.log('API connection test successful', 'success');
            return true;
        } catch (error) {
            this.log(`API connection test failed: ${error.message}`, 'error');
            return false;
        }
    }

    // Mise à jour du statut de connexion dans l'UI
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('api-status');
        if (statusElement) {
            statusElement.textContent = connected ? 'Connecté' : 'Déconnecté';
            statusElement.className = `status ${connected ? 'connected' : 'disconnected'}`;
        }

        const wsStatusElement = document.getElementById('ws-status');
        if (wsStatusElement) {
            wsStatusElement.textContent = connected ? 'Connecté' : 'Déconnecté';
            wsStatusElement.className = `status ${connected ? 'connected' : 'disconnected'}`;
        }
    }

    // Logger avec gestion UI
    log(message, type = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const logMessage = `[${timestamp}] ${message}`;
        
        console.log(logMessage);
        
        // Ajouter au log UI si disponible
        if (typeof addToActivityLog === 'function') {
            addToActivityLog(message, type);
        }
    }

    // Déconnexion propre
    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
        this.isConnected = false;
        this.subscriptions.clear();
        this.callbacks.clear();
        this.log('API disconnected', 'info');
    }
}

// === GESTIONNAIRE PRINCIPAL DE L'API ===

class APIManager {
    constructor() {
        this.bybitAPI = new BybitAPI();
        this.isInitialized = false;
        this.activeSymbol = 'BTCUSDT';
        this.monitoringInterval = null;
    }

    // Initialisation de l'API
    async initialize(apiKey, apiSecret, isTestnet = true) {
        try {
            this.bybitAPI.setCredentials(apiKey, apiSecret, isTestnet);
            
            // Test de connexion
            const connected = await this.bybitAPI.testConnection();
            if (!connected) {
                throw new Error('Failed to connect to Bybit API');
            }

            // Connexion WebSocket
            await this.bybitAPI.connectWebSocket();
            
            // Souscriptions de base
            this.setupSubscriptions();
            
            // Démarrage du monitoring
            this.startMonitoring();
            
            this.isInitialized = true;
            this.bybitAPI.log('API Manager initialized successfully', 'success');
            
            return true;
        } catch (error) {
            this.bybitAPI.log(`API initialization failed: ${error.message}`, 'error');
            return false;
        }
    }

    // Configuration des souscriptions WebSocket
    setupSubscriptions() {
        // Souscription aux données de prix
        this.bybitAPI.subscribe(`tickers.${this.activeSymbol}`, (data) => {
            this.handleTickerUpdate(data);
        });

        // Souscription aux positions (si authentifié)
        if (this.bybitAPI.apiKey) {
            this.bybitAPI.subscribe('position', (data) => {
                this.handlePositionUpdate(data);
            });
        }
    }

    // Gestion des mises à jour de prix
    handleTickerUpdate(data) {
        const ticker = data.data;
        
        // Mettre à jour l'interface de prix
        if (typeof updatePriceDisplay === 'function') {
            updatePriceDisplay(ticker);
        }
    }

    // Gestion des mises à jour de positions
    handlePositionUpdate(data) {
        // Mettre à jour l'interface des positions
        if (typeof updatePositionDisplay === 'function') {
            updatePositionDisplay(data.data);
        }
    }

    // Démarrage du monitoring périodique
    startMonitoring() {
        this.monitoringInterval = setInterval(async () => {
            try {
                // Récupération périodique des données de compte
                if (this.bybitAPI.apiKey) {
                    const balance = await this.bybitAPI.getAccountBalance();
                    if (typeof updateAccountBalance === 'function') {
                        updateAccountBalance(balance);
                    }
                }
            } catch (error) {
                this.bybitAPI.log(`Monitoring error: ${error.message}`, 'error');
            }
        }, 10000); // Toutes les 10 secondes
    }

    // Arrêt du monitoring
    stopMonitoring() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
        }
    }

    // Changement de symbole de trading
    changeSymbol(symbol) {
        // Désabonnement de l'ancien symbole
        this.bybitAPI.unsubscribe(`tickers.${this.activeSymbol}`);
        
        // Changement de symbole actif
        this.activeSymbol = symbol;
        
        // Souscription au nouveau symbole
        this.bybitAPI.subscribe(`tickers.${symbol}`, (data) => {
            this.handleTickerUpdate(data);
        });
        
        this.bybitAPI.log(`Active symbol changed to ${symbol}`, 'info');
    }

    // Exécution d'un trade
    async executeTrade(side, quantity, orderType = 'Market', price = null) {
        try {
            const result = await this.bybitAPI.placeOrder(
                this.activeSymbol,
                side,
                orderType,
                quantity,
                price
            );
            
            // Notification de succès
            if (typeof showNotification === 'function') {
                showNotification(`Ordre ${side} exécuté avec succès`, 'success');
            }
            
            return result;
        } catch (error) {
            // Notification d'erreur
            if (typeof showNotification === 'function') {
                showNotification(`Erreur lors de l'exécution: ${error.message}`, 'error');
            }
            throw error;
        }
    }

    // Fermeture de toutes les positions
    async closeAllPositions() {
        try {
            const positions = await this.bybitAPI.getPositions();
            const openPositions = positions.list.filter(p => parseFloat(p.size) !== 0);
            
            for (const position of openPositions) {
                await this.bybitAPI.closePosition(position.symbol, position.side);
            }
            
            if (typeof showNotification === 'function') {
                showNotification('Toutes les positions fermées', 'success');
            }
        } catch (error) {
            if (typeof showNotification === 'function') {
                showNotification(`Erreur fermeture positions: ${error.message}`, 'error');
            }
            throw error;
        }
    }

    // Arrêt propre
    shutdown() {
        this.stopMonitoring();
        this.bybitAPI.disconnect();
        this.isInitialized = false;
        this.bybitAPI.log('API Manager shut down', 'info');
    }
}

// Instance globale de l'API Manager
const apiManager = new APIManager();

// === FONCTIONS D'INTÉGRATION AVEC L'INTERFACE ===

// Fonction d'initialisation à appeler depuis l'interface
async function initializeAPI(apiKey, apiSecret, isTestnet = true) {
    return await apiManager.initialize(apiKey, apiSecret, isTestnet);
}

// Fonctions de trading à appeler depuis l'interface
async function buyMarket(quantity) {
    return await apiManager.executeTrade('Buy', quantity, 'Market');
}

async function sellMarket(quantity) {
    return await apiManager.executeTrade('Sell', quantity, 'Market');
}

async function buyLimit(quantity, price) {
    return await apiManager.executeTrade('Buy', quantity, 'Limit', price);
}

async function sellLimit(quantity, price) {
    return await apiManager.executeTrade('Sell', quantity, 'Limit', price);
}

// Export des classes pour utilisation modulaire
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BybitAPI, APIManager, apiManager };
}