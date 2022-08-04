/*
 * Author: Leonardo Kopeski
 * Last Update: 04/08/2022
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
    static toBinary(number, len){
        let binary = number.toString(2)
        while(binary.length < len){
            binary = "0" + binary
        }
        return binary
    }
}

class array2D{
    constructor(height, width, value){
        this.height = height
        this.width = width

        if(!value){
            this.data = []
            for(var x = 0; x < height; x++){
                var arr = []
                for(var y = 0; y < width; y++){
                    arr.push(templateFunctions.random(1))
                }
                this.data.push(arr)
            }
        }else{
            this.data = value
        }
    }

    map(callback){
        if(typeof this.data[0] != "object"){
            this.data = [this.data]
        }
        this.height = this.data.length
        this.width = this.data[0].length

        return this.data.map((arr, x)=>{
            return arr.map((num, y)=>{
                return callback(num, x, y)
            })
        })
    }

    static transpose(a){
        var arr = new array2D(a.width, a.height)
        arr.data = arr.map((arr, x, y)=>{
            return a.data[y][x] || a.data[y]
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
            var elm1 = a.data[x][y] || a.data[x]
            var elm2 = b.data[x][y] || b.data[x]
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

export class neuralNetwork{
    constructor(nodes, learningRate = .1){
        this.nodes = nodes
        this.bias = []
        this.weigths = []

        this.dataSet = {inputs: [], outputs: []}
        this.learningRate = learningRate

        for(var c = 1; c < nodes.length; c++){
            this.bias.push(new array2D(nodes[c], 1))
        }
        for(var c = 1; c < nodes.length; c++){
            this.weigths.push(new array2D(nodes[c], nodes[c-1]))
        }
    }

    set(input, expectedOutput){
        this.dataSet.inputs.push(input)
        this.dataSet.outputs.push(expectedOutput)
    }

    async asyncExecute(inputArr){
        var res = new array2D(this.inputNodes, 1, inputArr)
        for(var c = 0; c < this.nodes.length-1; c++){
            res = array2D.multiply(this.weigths[c], res)
            res = array2D.sum(res, this.bias[c])
            res.data = res.map(templateFunctions.sigmoid)
        }

        return array2D.transpose(res).data[0]
    }

    execute(inputArr){
        var res = new array2D(this.inputNodes, 1, inputArr)
        for(var c = 0; c < this.nodes.length-1; c++){
            res = array2D.multiply(this.weigths[c], res)
            res = array2D.sum(res, this.bias[c])
            res.data = res.map(templateFunctions.sigmoid)
        }

        return array2D.transpose(res).data[0]
    }

    async train(repeats){
        for(var e = 0; e < repeats; e++){
            for(var c in this.dataSet.inputs){
                var inputArr = this.dataSet.inputs[c]
                var outputArr = this.dataSet.outputs[c]

                var history = [new array2D(this.nodes[0], 1, inputArr)]
                var output = new array2D(this.inputNodes, 1, inputArr)

                // FeedForward
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
                    gradient = array2D.escalarMultiply(gradient, this.learningRate)
                    this.bias[c-1] = array2D.sum(this.bias[c-1], gradient)
        
                    var deltas = array2D.multiply(gradient, historyTransposed[c-1])
                    this.weigths[c-1] = array2D.sum(this.weigths[c-1], deltas)
                }
            }
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

        var res = new neuralNetwork(obj.nodes)
        res.weigths = obj.weigths.map(x => {
            return new array2D(x.length, x[0].length, x)
        })
        res.bias = obj.bias.map(x => {
            return new array2D(x.length, x[0].length, x)
        })

        return res
    }
}

export class wordIdentifier{
    constructor(maxIndex, wordEmbedding, prefab = null){
        if(maxIndex < 2){maxIndex = 2}
        this.maxIndexes = Math.ceil(Math.log(maxIndex) / Math.log(2))
        this.neuralNetwork = !prefab?
            new neuralNetwork([50, 5, this.maxIndexes, this.maxIndexes], .5) : 
            neuralNetwork.fromRecipe(prefab)
        this.wordEmbedding = wordEmbedding
    }
    useWordEmbedding(message){
        let embeddings = []
        for(var c in message.split(" ")){
            let word = message.split(" ")[c]
                .replaceAll(",", "")
                .replaceAll("!", "")
                .replaceAll("?", "")

            if(this.wordEmbedding[word]){
                embeddings.push(this.wordEmbedding[word])
            }
        }

        if(embeddings.length < 1){
            embeddings.push(Array(50).fill(0))
        }

        let sum = []
        for(var i = 0; i < embeddings.length; i++){
            for(var j = 0; j < embeddings[i].length; j++){
                if(!sum[j]){
                    sum[j] = 0
                }
                sum[j] += embeddings[i][j]
            }
        }

        let res = sum.map(x=>x/embeddings.length)

        return res
    }
    run(message){
        let input = this.useWordEmbedding(message)
        let rawResponse = this.neuralNetwork.execute(input)
        let response = rawResponse.map(x=>{
            return Math.round(x)
        })
        let decimal = parseInt(response.join(""), 2)
        return decimal
    }
    train({dataset, iterations}){
        Object.keys(dataset).forEach(message=>{
            let input = this.useWordEmbedding(message)
            
            let output = templateFunctions.toBinary(dataset[message], this.maxIndexes)
                .split("")
                .map(x=>parseInt(x))
            
            this.neuralNetwork.set(input, output)
        })
    
        this.neuralNetwork.train(iterations)
    }
}
