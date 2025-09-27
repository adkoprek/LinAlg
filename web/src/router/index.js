import { createRouter, createWebHistory } from 'vue-router';
import HomeView from '@/views/HomeView.vue';
import ManualView from '@/views/ManualView.vue';
import CodingView from '@/views/CodingView.vue';

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
        path: '/',
        name: 'home',
        component: HomeView
    },
    {
        path: '/manual',
        name: 'manual',
        component: ManualView 
    },
    {
      path: '/coding',
      name: 'coding',
      component: CodingView
    }
  ]
});

export default router;
