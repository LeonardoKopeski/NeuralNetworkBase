/*
 * Author: Leonardo Kopeski
 * Last Update: 7/01/2023
 */

class matrix2D{
    constructor(width, height, data){
        this.width = width
        this.height = height
        this.data = data
    }
    sigmoid(){
        var arr = this.data.map(subarr=>{
            return subarr.map((value)=>{
                return 1 / (1 + Math.exp(-value))
            })
        })
        return new matrix2D(this.width, this.height, arr)
    }
    derivatedSigmoid(){
        var arr = this.data.map(subarr=>{
            return subarr.map((value)=>{
                return value * (1 - value)
            })
        })
        return new matrix2D(this.width, this.height, arr)
    }
    print(){
        console.log(`width: ${this.width} | height: ${this.height}`)
        console.log(this.data)
        console.table(this.data)
    }
    static convertUndefined(value){
        return value == undefined? 1: value
    }
    static fromRandomValues(width, height){
        var arr = []
        for(let x = 0; x < width; x++){
            let subarr = []
            for(let y = 0; y < height; y++){
                subarr.push(Math.random()*2-1)
            }
            arr.push(subarr)
        }
        return new matrix2D(width, height, arr)
    }
    static layerMultiplication(matrix1, matrix2){
        var arr = matrix1.data[0].map((subarr, y)=>{
            var sum = 0
            for (let k = 0; k < matrix1.width; k++) {
                var elm1 = matrix2D.convertUndefined(matrix1.data[k][y])
                var elm2 = matrix2D.convertUndefined(matrix2.data[k][0])

                sum += elm1 * elm2
            }
            return sum
        })
        return new matrix2D(1, matrix1.height, [arr])
    }
    static layerSum(matrix1, matrix2){
        var arr = matrix1.data[0].map((subarr, x)=>{
            let sum = matrix2.data.map(value=>value[x]).reduce((a, b)=>a+b, 0)
            return [sum+matrix1.data[0][x]]
        })

        return new matrix2D(matrix1.height, matrix1.width, arr)
    }
    static dotMultiplication(matrix1, matrix2){
        var arr = []
        for(var x = 0; x < matrix2.width; x++){
            var subarray = []
            for(var y = 0; y < matrix1.height; y++){
                let sum = 0
                for (let k = 0; k < matrix1.width; k++) {
                    var elm1 = matrix2D.convertUndefined(matrix2.data[x][k])
                    var elm2 = matrix2D.convertUndefined(matrix1.data[k][y])
                    sum += elm1 * elm2
                }
                subarray.push(sum)
            }
            arr.push(subarray)
        }
        return new matrix2D(matrix2.width, matrix1.height, arr)
    }
    static crossMultiplication(matrix1, matrix2){
        var arr = matrix1.data.map((subarr, x)=>{
            return subarr.map((value, y)=>{
                return matrix1.data[x][y] * matrix2.data[x][y]
            })
        })
        return matrix2D.transpose(new matrix2D(matrix1.width, matrix1.height, arr))
    }
    static simpleSum(matrix1, matrix2){
        var arr = matrix1.data.map((subarr, x)=>{
            return subarr.map((value, y)=>{
                return matrix1.data[x][y] + matrix2.data[x][y]
            })
        })
        return new matrix2D(matrix1.width, matrix1.height, arr)
    }
    static scaleMultiplication(matrix1, scale){
        var arr = matrix1.data.map((subarr, x)=>{
            return subarr.map((value, y)=>{
                return matrix1.data[x][y] * scale
            })
        })
        return new matrix2D(matrix1.width, matrix1.height, arr)
    }
    static sum(matrix1, matrix2){
        var arr = matrix2.data.map((subarr, x)=>{
            return subarr.map((value, y)=>{
                return matrix1.data[x][y] + matrix2.data[x][y]
            })
        })
        return new matrix2D(matrix2.width, matrix2.height, arr)
    }
    static subtract(matrix1, matrix2){
        var arr = matrix1.data.map((subarr, x)=>{
            return subarr.map((value, y)=>{
                return matrix1.data[x][y] - matrix2.data[x][y]
            })
        })
        return new matrix2D(matrix1.width, matrix1.height, arr)
    }
    static sumError(matrix1, matrix2){
        var sum = 0
        for(var x = 0; x < matrix1.width; x++){
            for(var y = 0; y < matrix1.height; y++){
                sum += Math.abs(matrix1.data[x][y] - matrix2.data[x][y])
            }
        }
        return sum
    }
    static transpose(matrix){
        var arr = matrix2D.returnTransposedArray(matrix.data)
        return new matrix2D(matrix.height, matrix.width, arr)
    }
    static returnTransposedArray(data){
        var arr = []
        for(let x = 0; x < data[0].length; x++){
            let subarr = []
            for(let y = 0; y < data.length; y++){
                subarr.push(data[y][x])
            }
            arr.push(subarr)
        }
        return arr
    }
}

class dataset{
    constructor(){
        this.inputs = []
        this.outputs = []
    }
    add(input, output){
        this.inputs.push(input)
        this.outputs.push(output)
    }
}

class neuralNetwork{
    constructor(nodes, learningRate = .1){
        this.nodes = nodes
        this.learningRate = learningRate

        this.bias = []
        this.weigths = []
        for(var c = 1; c < nodes.length; c++){
            this.bias.push(matrix2D.fromRandomValues(1, nodes[c]))
        }
        for(var c = 1; c < nodes.length; c++){
            this.weigths.push(matrix2D.fromRandomValues(nodes[c-1], nodes[c]))
        }
    }
    execute(input){
        var value = new matrix2D(this.nodes[0], 1, matrix2D.returnTransposedArray([input]))
        for(var c = 0; c < this.nodes.length-1; c++){
            value = matrix2D.layerMultiplication(this.weigths[c], value)
            value = matrix2D.layerSum(this.bias[c], value)
            value = value.sigmoid()
        }

        return matrix2D.transpose(value).data[0]
    }
    train(dataset, repeats){
        for(let r = 0; r < repeats;r++){
            for(let d = 0; d < dataset.inputs.length; d++){
                let inputArr = matrix2D.returnTransposedArray([dataset.inputs[d]])
                let outputArr = matrix2D.returnTransposedArray([dataset.outputs[d]])

                let output = new matrix2D(this.nodes[0], 1, inputArr)
                let history = [output]

                // FeedForward
                for(let c = 0; c < this.nodes.length-1; c++){
                    output = matrix2D.layerMultiplication(this.weigths[c], output)
                    output = matrix2D.layerSum(this.bias[c], output)
                    output = output.sigmoid()

                    history.push(output)
                }

                // BackPropagation
                let historyDerivated = history.map(elm => matrix2D.transpose(elm.derivatedSigmoid()))
                
                let expected = new matrix2D(this.nodes[this.nodes.length-1], 1, outputArr)
                let error = matrix2D.transpose(matrix2D.subtract(expected, output))
                for(let c = this.nodes.length-1; c >= 1;c--){

                    let gradient = matrix2D.crossMultiplication(error, historyDerivated[c])
                    gradient = matrix2D.scaleMultiplication(gradient, this.learningRate)
                    gradient = matrix2D.transpose(gradient)
                    this.bias[c-1] = matrix2D.simpleSum(this.bias[c-1], gradient)

                    let deltas = matrix2D.dotMultiplication(gradient, history[c-1])
                    this.weigths[c-1] = matrix2D.sum(this.weigths[c-1], deltas)

                    if(c == 0) break
                    let nextWeigth = matrix2D.transpose(this.weigths[c-1])
                    error = matrix2D.dotMultiplication(nextWeigth, error)
                }
            }
        }
    }

    trainUntilScore(dataset, percentage){
        let repeats = 0
        let errorMax = dataset.inputs.length * this.nodes[this.nodes.length-1]
        let score = 0
        while(score < percentage) {
            repeats++
            let errorSum = 0
            for(let d = 0; d < dataset.inputs.length; d++){
                let inputArr = matrix2D.returnTransposedArray([dataset.inputs[d]])
                let outputArr = matrix2D.returnTransposedArray([dataset.outputs[d]])

                let output = new matrix2D(this.nodes[0], 1, inputArr)
                let history = [output]

                // FeedForward
                for(let c = 0; c < this.nodes.length-1; c++){
                    output = matrix2D.layerMultiplication(this.weigths[c], output)
                    output = matrix2D.layerSum(this.bias[c], output)
                    output = output.sigmoid()

                    history.push(output)
                }

                // BackPropagation
                let historyDerivated = history.map(elm => matrix2D.transpose(elm.derivatedSigmoid()))
                
                let expected = new matrix2D(this.nodes[this.nodes.length-1], 1, outputArr)
                let error = matrix2D.subtract(expected, output)

                errorSum += matrix2D.sumError(expected, output)

                for(let c = this.nodes.length-1; c >= 1;c--){
                    let gradient = matrix2D.crossMultiplication(error, historyDerivated[c])
                    gradient = matrix2D.scaleMultiplication(gradient, this.learningRate)
                    gradient = matrix2D.transpose(gradient)
                    this.bias[c-1] = matrix2D.simpleSum(this.bias[c-1], gradient)

                    let deltas = matrix2D.dotMultiplication(gradient, history[c-1])
                    this.weigths[c-1] = matrix2D.sum(this.weigths[c-1], deltas)

                    if(c == 0) break
                    let nextWeigth = matrix2D.transpose(this.weigths[c-1])
                    error = matrix2D.dotMultiplication(nextWeigth, error)
                }
            }

            score = 100 + errorSum / errorMax * -200
        }
        return repeats
    }
    
    generateRecipe(){
        return JSON.stringify({
            nodes: this.nodes,
            weigths: this.weigths.map(x => x.data),
            bias: this.bias.map(x => x.data)
        })
    }

    static mutate(base, learningRate = .3){
        var n = new neuralNetwork(base.nodes)
        n.weigths = base.weigths.map(w => {
            return array2D.fromValue(w.map((value)=>value * (1 + templateFunctions.random(learningRate))))
        })
        n.bias = base.bias.map(b => {
            return array2D.fromValue(b.map((value)=>value * (1 + templateFunctions.random(learningRate))))
        })

        return n
    }

    static fromRecipe(recipe, legacy = false){
        var obj = JSON.parse(recipe)

        var res = new neuralNetwork(obj.nodes)
        res.weigths = obj.weigths.map(x => {
            return legacy?
                matrix2D.transpose(new matrix2D(x.length, x[0].length, x)):
                new matrix2D(x.length, x[0].length, x)
        })
        res.bias = obj.bias.map(x => {matrix2D
            return legacy?
                matrix2D.transpose(new matrix2D(x.length, x[0].length, x)):
                new matrix2D(x.length, x[0].length, x)
        })

        return res
    }
}

module.exports = { neuralNetwork, dataset }
