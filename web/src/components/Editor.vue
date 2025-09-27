<script setup>
import { ref, onMounted } from "vue";
import { CodeEditor } from "monaco-editor-vue3";
import "monaco-editor-vue3/dist/style.css";
import { loadPyodide } from "pyodide";
import { VueSpinnerIos } from "vue3-spinners";
import * as monaco from "monaco-editor";
import DragRow from "vue-resizer/DragRow.vue";

const code = ref(`print("Hello from Python!")`);
const output = ref("");
const isRunning = ref(false);
const isLoading = ref(true);
let pyodide = null;

async function initPyodide() {
  pyodide = await loadPyodide();
  console.log("Pyodide initialized!");
  isLoading.value = false;
  pyodide.setStdout({
    batched: (msg) => {
      output.value += msg;
    },
  });
  pyodide.setStderr({
    batched: (msg) => {
      output.value += "Error: " + msg;
    },
  });
}

onMounted(async () => {
  await initPyodide();
  monaco.editor.defineTheme("my", {
    base: 'vs',
    inherit: true,
    rules: [],
    colors: {
      'editor.background': '#ffffff'
    }
  });
});

async function runCode() {
  output.value = "";
  isRunning.value = true;
  try {
    const result = await pyodide.runPythonAsync(code.value);
    if (result !== undefined) {
      output.value += result + "\n";
    }
  } catch (err) {
    output.value += "Error: " + err.message + "\n";
  } finally {
    isRunning.value = false;
    const term = document.querySelector("#terminal");
    if (term) term.scrollTop = term.scrollHeight;
  }
}

const editorOptions = {
  fontSize: 12,
  minimap: { enabled: false },
  automaticLayout: true,
};
</script>

<template>
  <div class="w-full h-full flex pr-6 pl-2 py-6">
    <div
      class="w-full bg-white rounded-2xl shadow-xl p-6 flex flex-col">
      <div
        v-if="isLoading"
        class="flex flex-col items-center justify-center flex-1 space-y-4">
        <VueSpinnerIos size="50" color="#4F46E5" />
        <div class="text-gray-600 text-lg font-medium">
          Loading Python environment...
        </div>
      </div>

      <div v-else class="flex flex-col flex-1 h-screen">
        <DragRow class="flex-1" slider-bg-color="transparent" slider-bg-hover-color="trasparent" style="width: 100%;">
          <template #top>
            <div class="h-full overflow-hidden rounded-xl border shadow-inner">
              <CodeEditor
                v-model:value="code"
                language="python"
                theme="my"
                :options="editorOptions"
              />
            </div>
          </template>
          <template #bottom>
            <div 
              id="terminal"
              class="h-full bg-black text-green-400 font-mono p-4 rounded-xl shadow-inner overflow-y-auto whitespace-pre-wrap">
              {{ output }}
            </div>
          </template>
        </DragRow>
        
        <div class="flex justify-end mt-3">
          <button
            @click="runCode"
            class="px-6 py-2 bg-indigo-600 text-white rounded-xl shadow hover:bg-indigo-700 disabled:opacity-50 transition"
            :disabled="isRunning">
            {{ isRunning ? "Running..." : "Run Code" }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>