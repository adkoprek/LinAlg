<script setup>
import Hero from '@/components/Hero.vue';
import Roadmap from '@/components/Roadmap.vue';
import Footer from '@/components/Footer.vue';
import About from '@/components/About.vue';
</script>


<template>
  <div class="min-h-screen flex flex-col bg-gradient-to-b from-slate-50 to-slate-100 text-slate-800">
    <Hero />
    <Roadmap />
    <About />
    <Footer />
  </div>
</template>
