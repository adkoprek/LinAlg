<script setup>
const props = defineProps({
  title: String,
  description: String,
});
</script>

<template>
  <div class="w-full flex justify-center pl-6 pr-2 py-6 h-full">
    <div
      class="w-full bg-white rounded-2xl shadow-xl p-6 flex flex-col">
    <h2 class="text-4xl font-bold">{{ title }}</h2>
    <div v-katex="description" class="text-lg text-gray-700"></div>
    </div>
  </div>
</template>
