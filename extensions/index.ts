// pi-package entry: loads the built extension from dist/.
// (pi loads extensions/ directly; the implementation lives in src/extension, typechecked + built.)
import activate from "../dist/extension/index.js";

export default activate;
