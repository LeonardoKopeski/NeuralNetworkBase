/*
 * Author: Leonardo Kopeski
 * Last Update: 27/09/2021
 */

class templateFunctions{
    static sigmoid(x){
        return 1 / (1 + Math.exp(-x))
    }
    static dsigmoid(x){
        return x * (1 - x)
    }
    static random(x){
        return (Math.random()*(x*2))-x
    }
    static validateDataSet(dataSet){
        if(dataSet.inputs == undefined){
            return false
        }
        if(dataSet.outputs == undefined){
            return false
        }
        if(dataSet.inputs.length != dataSet.outputs.length){
            return false
        }
        var ok = true
        dataSet.inputs.forEach(elm=>{
            if(typeof elm != "object" || elm[0] == undefined){
                ok = false
            }
        })
        if(!ok){
            return false
        }

        return true
    }
}

class array2D{
    constructor(height, width, value){
        this.height = height
        this.width = width

        if(value){
            this.data = value
        }else{
            this.data = []
            for(var x = 0; x < height; x++){
                var arr = []
                for(var y = 0; y < width; y++){
                    arr.push(templateFunctions.random(1))
                }
                this.data.push(arr)
            }
        }
    }

    retify(){
        if(typeof this.data[0] != "object"){
            this.data = [this.data]
        }
        this.height = this.data.length
        this.width = this.data[0].length
    }

    map(callback){
        this.retify()

        return this.data.map((arr, x)=>{
            return arr.map((num, y)=>{
                return callback(num, x, y)
            })
        })
    }

    toArray(){
        return this.data
    }

    static transpose(a){
        var arr = new array2D(a.width, a.height)
        arr.data = arr.map((arr, x, y)=>{
            return a.data[y][x] ? a.data[y][x] : a.data[y]
        })
        return arr
    }

    static hadamard(a, b){
        var arr = new array2D(a.height, a.width)
        arr.data = arr.map((arr, x, y)=>{
            return a.data[x][y] * b.data[x][y]
        })
        return arr
    }

    static sub(a, b){
        var arr = new array2D(a.height, a.width)
        arr.data = arr.map((arr, x, y)=>{
            var elm1 = a.data[x][y] == undefined? a.data[x] : a.data[x][y]
            var elm2 = b.data[x][y] == undefined? b.data[x] : b.data[x][y]
            return elm1 - elm2
        })
        return arr
    }

    static sum(a, b){
        var arr = new array2D(b.height, b.width)
        arr.data = arr.map((arr, x, y)=>{
            return a.data[x][y] + b.data[x][y]
        })
        return arr
    }

    static escalarMultiply(a, escalar){
        var arr = new array2D(a.height, a.width)
        arr.data = arr.map((arr, x, y)=>{
            return a.data[x][y] * escalar
        })
        return arr
    }

    static multiply(a, b) {
        var arr = new array2D(a.height, b.width)

        arr.data = arr.map((arr, x, y) => {
            let sum = 0
            for (let k = 0; k < a.width; k++) {
                var elm1 = a.data[x].length? a.data[x][k] : a.data[x]
                var elm2 = b.data[k].length? b.data[k][y] : b.data[k]
                sum += elm1 * elm2
            }
            return sum
        })

        return arr
    }
}

class neuron{
    constructor(nodes){
        this.nodes = nodes
        this.bias = []
        this.weigths = []

        for(var c = 1; c < nodes.length; c++){
            this.bias.push(new array2D(nodes[c], 1))
        }
        for(var c = 1; c < nodes.length; c++){
            this.weigths.push(new array2D(nodes[c], nodes[c-1]))
        }   
    }

    execute(inputArr){
        var res = new array2D(this.inputNodes, 1, inputArr)
        for(var c = 0; c < this.nodes.length-1; c++){
            res = array2D.multiply(this.weigths[c], res)
            res = array2D.sum(res, this.bias[c])
            res.data = res.map(templateFunctions.sigmoid)
        }

        return array2D.transpose(res).toArray()[0]
    }

    repeatedTrain(dataSet, executions, learningRate = .1){
        if(!templateFunctions.validateDataSet(dataSet)){
            throw "NeuralNetworkLibrary: Invalid DataSet"
        }

        for(var i = 0; i < executions; i++){
            for(var c in dataSet.inputs){
                this.train(dataSet.inputs[c], dataSet.outputs[c], learningRate)
            }
        }
    }

    train(inputArr, outputArr, learningRate = .1){
        // FeedForward
        var history = [new array2D(this.nodes[0], 1, inputArr)]
        var output = new array2D(this.inputNodes, 1, inputArr)
        for(var c = 0; c < this.nodes.length-1; c++){
            output = array2D.multiply(this.weigths[c], output)
            output = array2D.sum(output, this.bias[c])
            output.data = output.map(templateFunctions.sigmoid)
            
            history.push(output)
        }

        // BackPropagation
        var expected = new array2D(this.nodes[this.nodes.length-1], 1, outputArr)
        
        var historyTransposed = history.map(elm => array2D.transpose(elm))
        var historyDerivated = history.map(elm => new array2D(elm.height, elm.width, elm.map(templateFunctions.dsigmoid)))

        var error = array2D.sub(expected, output)
        for(var c = this.nodes.length-1; c >= 1;c--){
            if(c != this.nodes.length-1){
                var nextWeigth = array2D.transpose(this.weigths[c])
                error = array2D.multiply(nextWeigth, error)
            }

            var gradient = array2D.hadamard(error, historyDerivated[c])
            gradient = array2D.escalarMultiply(gradient, learningRate)
            this.bias[c-1] = array2D.sum(this.bias[c-1], gradient)

            var deltas = array2D.multiply(gradient, historyTransposed[c-1])
            this.weigths[c-1] = array2D.sum(this.weigths[c-1], deltas)
        }
    }

    generateRecipe(){
        return JSON.stringify({
            nodes: this.nodes,
            weigths: this.weigths.map(x => x.data),
            bias: this.bias.map(x => x.data)
        })
    }

    static fromRecipe(recipe){
        var obj = JSON.parse(recipe)

        var res = new neuron(obj.nodes)
        res.weigths = obj.weigths.map(x => {
            return new array2D(x.length, x[0].length, x)
        })
        res.bias = obj.bias.map(x => {
            return new array2D(x.length, x[0].length, x)
        })

        return res
    }
}

module.exports = neuron