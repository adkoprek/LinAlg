<script setup>
import CodeInfoBlock from '@/components/CodeInfoBlock.vue';
import Footer from '@/components/Footer.vue';
import { RouterLink } from 'vue-router';
</script>


<template>
    <div class="bg-gray-50">
        <header class="flex flex-col items-center text-center pt-20 px-6">
            <h1 class="text-6xl font-extrabold mb-8 leading-tight max-w-4xl">
                Manual
            </h1>
            <p class="text-xl max-w-3xl mb-12 text-slate-600">
                Please read this before you start coding.
            </p>
        </header>
        <CodeInfoBlock/>
        <div class="w-full flex justify-center">
            <RouterLink to="/coding">
                <button class="px-12 py-5 bg-indigo-600 text-white text-xl font-semibold rounded-2xl shadow-lg hover:bg-indigo-700 transition transform hover:scale-105">
                    Start Coding
                </button>
            </RouterLink>
        </div>
        <Footer />
    </div>
</template>