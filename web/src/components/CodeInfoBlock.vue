<script setup>
import { onMounted } from "vue"
import Prism from "prismjs"
import "prismjs/components/prism-python"
import "prismjs/themes/prism.css"
import { ref } from 'vue';

onMounted(() => {
  Prism.highlightAll()
})

const manual_blocks = ref([
    {
        title: "Types",
        desc: `
            In your library you will be using premade custom types. 
            These types include mat and vec as shown in the example. 
            The mat type is a 2d-pthon list where every list is one row of a matrix
            and the vetor is just a simple list of floating point numbers.`,
        code: `
mat = list[list[float]]
vec = list[float]`
    },
    {
        title: "Errors",
        desc: `
            Your library should handle basic errors which will be tested. 
            The custom errors made for this library are: 
              • ShapeMismatchedError, when a shape of a matrix does not match
              • SingularError, when a matrix is singular to machiene precision
              • MaxIterationsError, when you exceed the amout of iterations when computing eigenvalues
            Do not worry if you are not familliar with this terms, just remember where you can look them up
        `,
        code: `
raise ShapeMismatchedError("Your custom message")
raise SingularError("Your custom message")
raise MaxIterationError("Your custom message")
        `
    },
    {
      title: "Function Setup",
      desc: `
        In every task you will get a short descripton, examples and theory. 
        In addition to that you will be told which parameters your function gets and which are expected to be returned.
        Then you write your code into the already provided empty function.
        Do not change the the function!
      `,
      code: `
def lu(a: mat) -> tuple[mat, mat, mat]:
  # Do your awesome PA=LU decomposition

  return P, L, U
      `
    },
    {
      title: "Array references",
      desc: `
        Be aware that your function must not change the parameters it recieves which will be checked.
        Most of the inputs are python lists which are passed by referenc which means that you will cahnge the original matrix.
        If would like to do changes on your matrix use the provided copy() function.
      `,
      code: `
def lu(a: mat) -> tuple[mat, mat, mat]:
      a_copy = a          # Do not do this
      a_copy[0][0] = 0    # This will change a 

      a_copy = copy(a)    # Do this 
      a_copy[0][0] = 0     
      `
    },
    {
      title: "Reusability",
      desc: `
        In every function you can reuse the function that you have already implemented otherwise no other libraries are available to you.
      `,
      code: `
def det(a: mat) -> float:
      P, L, U = lu(a)     # This is all right

      # Compute the determinant usign P, L, U

      return det
      `
    }
]);

</script>

<template>
  <section class="px-6 md:px-20 lg:px-40 py-12 md:py-20 bg-gray-50">
    <div 
      v-for="block in manual_blocks" 
      :key="block.title"
      class="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-10 items-start mb-16"
    >
      <h3 class="col-span-1 md:col-span-2 text-2xl md:text-3xl font-bold">
        {{ block.title }}
      </h3>

      <p class="text-gray-700 text-base md:text-lg leading-relaxed">
        {{ block.desc }}
      </p>

      <div>
        <pre class="rounded-lg shadow p-4 overflow-x-auto bg-white">
          <code class="language-python">{{ block.code }}</code>
        </pre>
      </div>
    </div>
  </section>
</template>
