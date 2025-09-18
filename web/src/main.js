import './assets/main.css'
import 'primeicons/primeicons.css'
import router from './router'

import { createApp } from 'vue'
import App from './App.vue'

let app = createApp(App)
app.use(router)
app.mount('#app')
