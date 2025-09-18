<script setup>
import logo from '@/assets/logo.png'
import { RouterLink, useRoute } from 'vue-router';

const isActiveLink = (routePath) => {
  const route = useRoute();
  return route.path == routePath;
}

</script>

<template>
    <nav class="flex justify-between items-center px-12 py-3 shadow bg-white sticky top-0 z-50">
      <div class="flex items-center space-x-3">
        <div class="w-8 h-8 rounded-xl">
            <img :src="logo" alt="Logo" class="w-full h-full object-cover" />
        </div>
        <span class="font-bold text-2xl">LinCode</span>
      </div>

      <div class="space-x-8 font-medium text-lg">
        <RouterLink
            to="/"
            :class="isActiveLink('/') ? 'text-indigo-600' : 'hover:text-indigo-600'"
        >Home</RouterLink>
        <RouterLink
            to="/tutorial"
            class="hover:text-indigo-600"
        >Manual</RouterLink>
        <RouterLink
            to="/roadmap"
            class="hover:text-indigo-600"
        >Coding</RouterLink>
      </div>
    </nav>
</template>
