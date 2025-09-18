<script>
import katexDirective from "@/katex-directive.js";

export default {
  name: "Roadmap",
  directives: { katex: katexDirective },
  data() {
    return {
      topics: [
        {
          title: "Vectors",
          desc: "Implement function for matnipulating vector and computing dot products.",
          latex: "\\vec{v} = \\begin{bmatrix} 1 \\\\ 2 \\\\ 3 \\end{bmatrix}",
        },
        {
          title: "Matrices",
          desc: "Learn how do code up matrix operations.",
          latex: "A = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}",
        },
        {
          title: "Matrix-Vector Operations",
          desc: "Combine your vector and matrix logic to create cross operations.",
          latex: "A\\vec{x} = \\vec{b}",
        },
        {
          title: "LU Decomposition",
          desc: "Implement a stable LU algorithem and solve linear equations.",
          latex: "A = LU",
        },
        {
          title: "Inverse",
          desc: "With yout LU code invert a matrix",
          latex: "A^{-1}A = I",
        },
        {
          title: "RREF",
          desc: "Implement the rref function that is not even implemented in numpy!",
          latex: "\\begin{bmatrix} 1 & 0 \\\\ 0 & 1 \\end{bmatrix}",
        },
        {
          title: "QR Decomposition",
          desc: "Implement the QR algorithem and learn householder reflections.",
          latex: "A = QR",
        },
        {
          title: "Determinant",
          desc: "Compute the determinant using the PA=LU decomposition.",
          latex: "\\det(A) = ad - bc",
        },
        {
          title: "Eigenvalues",
          desc: "Implement the brilliant eigenvalue algorithem. The eigenvectors are literally to complex.",
          latex: "A\\vec{v} = \\lambda\\vec{v}",
        },
      ],
    };
  },
};
</script>


<template>
  <section class="py-20 bg-gray-50">
    <div class="max-w-6xl mx-auto px-6">
      <h2 class="text-6xl font-bold text-center mb-12">Roadmap</h2>
      
      <div class="space-y-8">
        <div
          v-for="topic in topics"
          :key="topic.title"
          class="bg-white rounded-2xl shadow p-8 flex flex-col md:flex-row items-center justify-between hover:shadow-xl transition"
        >
          <!-- Left: Title + description -->
          <div class="md:w-1/2 mb-6 md:mb-0">
            <h3 class="text-2xl font-semibold mb-2">{{ topic.title }}</h3>
            <p class="text-gray-600">{{ topic.desc }}</p>
          </div>

          <!-- Right: LaTeX -->
          <div class="md:w-1/2 flex justify-center text-3xl">
            <span v-katex="topic.latex"></span>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
