<script setup>
import Task from '@/components/Task.vue';
import Editor from '@/components/Editor.vue';
import DragCol from "vue-resizer/DragCol.vue";
</script>


<template>
    <div class="w-full bg-gray-50">
        <DragCol style="width: 100vw; height: calc(100vh - 3.5rem);" slider-bg-color="transparent" slider-bg-hover-color="transparent">
            <template #left>
                <Task title="Test" />
            </template>
            <template #right>
                <Editor />
            </template>
        </DragCol>
    </div>
</template>