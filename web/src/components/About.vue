<script setup>
import profile from '@/assets/profile.png';
</script>

<template>
  <section class="py-20 bg-gray-50">
    <div class="border-b border-gray-800 border-1 max-w-5xl mx-auto"></div>
    <div class="max-w-5xl mx-auto px-6 mt-12 flex flex-col md:flex-row items-center gap-12">
      
      <div class="flex-shrink-0">
        <img 
          :src="profile" 
          alt="Adam Koprek" 
          class="w-60 h-60 rounded-full object-cover shadow-lg"
        />
      </div>
      
      <div>
        <h2 class="text-3xl font-bold mb-4">About Me</h2>
        <h3 class="text-xl font-semibold text-gray-700 mb-2">Adam Koprek</h3>
        <p class="text-gray-600 leading-relaxed">

            I'm a high school student and enjoy programming and math. 
            In my first high school year I noticed that most advanced programming uses linear algebra.
            So I took the 18.06 course from MIT taught by Gilbert Strang and started grinding.
            After I have mastered the theory I decided that I should learn it by implementing a linear algebra library.
            So I decided to make this library public on this website because there is so much to learn...
        </p>
      </div>
    </div>
  </section>
</template>