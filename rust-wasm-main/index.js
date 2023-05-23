import init, { Wrapper, new_wrapper_from_string} from './pkg/rust_wasm.js'

await init()

const n_max_iter = 10000000
const verbose = true

// Leitura de arquivo
function readTextFile(file) {
    let rawFile = new XMLHttpRequest();
    rawFile.open("GET", file, false);

    let fileContent = "";

    rawFile.onreadystatechange = function () {
        if (rawFile.readyState === 4) {
            if (rawFile.status === 200 || rawFile.status == 0) {
                fileContent = rawFile.responseText;
            }
        }
    }

    rawFile.send(null);

    return fileContent;
}

const params = []

for (let i = 0; i < 34; ++i) {
    params.push(i)
}

const answ_index = 2
const data = readTextFile('ionosphere.data')
// new_wrapper_from_string(data_str: String, pop_len: usize, n_params: usize, answ_index: usize, n_max_iter: usize, step_size: f64)
let wrapper = new_wrapper_from_string(data, 100, 34, 34, 100, 0.01)


console.log(wrapper.get_wrapper())

// console.log(wrapper.get_wrapper())
// train_wrapper(n_gen: usize, mutation_rate: f64, )
let init_time = Date.now()
const info = wrapper.train_wrapper(50, 0.1, 0.6)
let end_time = Date.now()
console.log(info)
console.log('Demorou ' + (end_time - init_time) + 'ms')

console.log(wrapper.get_wrapper())

// console.log(wrapper.get_wrapper())