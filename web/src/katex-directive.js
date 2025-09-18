import katex from "katex";
import "katex/dist/katex.min.css";

export default {
  mounted(el, binding) {
    katex.render(binding.value, el, {
      throwOnError: false,
      displayMode: false,
    });
  },
  updated(el, binding) {
    katex.render(binding.value, el, {
      throwOnError: false,
      displayMode: false,
    });
  },
};
