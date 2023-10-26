import Optimizer from "./optimizer";
import Model from "./model";
import Compiler from "./compiler";

export default interface Options {
    model: Model;
    inputs: string[];
    outputs: string[];
    optimizer: Optimizer;
    compiler: Compiler;
    transforms?: {
        resizeImage: {
            width: number;
            height: number;
            depth: number;
        }
    }
}