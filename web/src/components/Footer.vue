<template>
  <footer class="bg-gray-900 text-gray-200 py-12 mt-20">
    <div class="flex justify-around px-6">
      <div>
        <h2 class="text-2xl font-bold text-white mb-3">LinCode</h2>
        <p class="text-gray-400">
          Practice linear algebra concepts interactively, just like coding challenges.
        </p>
      </div>

      <div>
        <h3 class="text-lg font-semibold mb-3 text-white">Contact</h3>
        <ul class="space-y-2">
          <li>Email: <a href="mailto:adam.koprek@stud.kswe.ch" class="hover:text-white">youremail@example.com</a></li>
          <li>GitHub: <a href="https://github.com/adkoprek" target="_blank" class="hover:text-white">github.com/yourgithub</a></li>
        </ul>
      </div>
    </div>

    <div class="border-t border-gray-700 mt-10 pt-6 text-center text-gray-500 text-sm">
      © {{ new Date().getFullYear() }} LinCode. All rights reserved.
    </div>
  </footer>
</template>
