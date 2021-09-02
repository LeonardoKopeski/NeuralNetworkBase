class neuralNetwork{
    constructor(nodes){
        this.nodes = nodes

        this.weigths = {}
        for(var i = 0 ; i < nodes.length-1 ; i++){
            this.weigths[i] = {}
            for(var j = 0; j < nodes[i]; j++){
                for(var k = 0; k < nodes[i+1]; k++){
                    this.weigths[i][`${j}-${k}`] = Math.random()*2-1
                }
            }
        }

        this.bias = {}
        for(var i = 1 ; i < nodes.length ; i++){
            this.bias[i] = {}
            for(var j = 0; j < nodes[i]; j++){
                this.bias[i][j] = Math.random()*2-1
            } 
        }
    }

    train(inputArray, expectedOutput, learningRate){
        //formula = peso*lr*(derivative/peso)
        var res = this.execute(inputArray)
        var err = {}
        
        this.nodes.forEach((a, i)=>{
            err[i] = {}
        })

        res.forEach((elm, index)=>{
            err[0][index] = elm - expectedOutput[index]
        })

        for(var i = this.nodes.length-2; i >= 0; i--){
            var arr = []
            var invertedIndex = Math.abs(i-(this.nodes.length-2))

            Object.keys(err[invertedIndex]).forEach(elm=>{
                for(var c = 0; c < this.nodes[i]; c++){
                    var weigth = this.weigths[i][c+"-"+elm]
                    err[invertedIndex+1][c] = elm * weigth
                }
            })
        }

        Object.keys(err).forEach(x => {
            this.weigths
        })

        return Object.values(res)

        /*
        erro = (mx + b - num)^2

        dm = e * x * m'
        db = e

        dw = eºd(s) * lr * 0
        */
    }

    execute(inputArray){
        var res = Object.assign({}, inputArray)

        this.nodes.forEach((node, nodeIndex) => {
            var update = {}
            if(this.weigths[nodeIndex] != undefined){
                Object.keys(this.weigths[nodeIndex]).forEach(k => {
                    var path = k.split("-")
                    var m = this.weigths[nodeIndex][k]

                    if(update[path[1]] == undefined){
                        update[path[1]] = 0
                    }

                    update[path[1]] += res[path[0]] * m
                })
            }

            if(this.bias[nodeIndex] != undefined){
                Object.values(update).forEach((x, i) => {
                    update[i] += this.bias[nodeIndex][i]
                })
                update = Object.assign({}, Object.values(update).map(this.sigmoid))
            }

            if(update[0] == undefined){
                return
            }
            res = {...update}
        })

        return Object.values(res)
    }

    derivative(x, dx) {
        dx = .0000001;
        return (Math.pow(x+dx, 2) - Math.pow(x, 2)) / dx
    }

    percentage(p, v){
        return p/100*v
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    dsigmoid(x){
        return x * (1-x); 
    }
}

module.exports = {
    neuralNetwork
}