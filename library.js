/*
 * Author: Leonardo Kopeski
 * Last Update: 06/09/2021
 */

class array2D{
    constructor(height,width,value=null){
        this.height = height
        this.width = width
        if(value == null){
            this.data = []
            for(var x = 0; x < height; x++){
                var arr = []
                for(var y = 0; y < width; y++){
                    arr.push(Math.random()*2-1)
                }
                this.data.push(arr)
            }
        }else{
            this.data = value
        }
    }

    print(){
        console.table(this.data)
    }

    map(func,data = this.data){
        if(typeof this.data[0] != "object"){
            this.data = [this.data]
        }
        this.height = this.data.length
        this.width = this.data[0].length

        var newData = data.map((arr,i)=>{
            return data[i].map((num,j)=>{
                return func(num,i,j)
            })
        })
        return newData
    }

    toArray(){
        return this.data
    }

    static transpose(a){
        var arr = new array2D(a.width,a.height)
        arr.data = arr.map((arr,i,j)=>{
            var r
            if(a.data[j][i] == undefined){
                r = a.data[j]
            }else{
                r = a.data[j][i]
            }
            return r
        })
        return arr
    }

    static hadamard(a,b){
        var arr = new array2D(a.height,a.width)
        arr.data = arr.map((arr,i,j)=>{
            return a.data[i][j]*b.data[i][j]
        })
        return arr
    }

    static sub(a,b){
        var arr = new array2D(a.height,a.width)
        arr.data = arr.map((arr,i,j)=>{
            var r1
            var r2
            if(a.data[i][j] == undefined){
                r1 = a.data[i]
            }else{
                r1 = a.data[i][j]
            }
            if(b.data[i][j] == undefined){
                r2 = b.data[i]
            }else{
                r2 = b.data[i][j]
            }
            return r1 - r2
        })
        return arr
    }

    static sum(a,b){
        var arr = new array2D(a.height,a.width)
        arr.data = arr.map((arr,i,j)=>{
            var r1
            var r2
            try{
                r1 = a.data[i][j]
            }catch{
                r1 = a.data[i]
            }
            try{
                r2 = b.data[i][j]
            }catch{
                r2 = b.data[i]
            }
            return r1 + r2
        })
        return arr
    }

    static escalarMultiply(a,escalar){
        var arr = new array2D(a.height,a.width)
        arr.data = arr.map((arr,i,j)=>{
            return a.data[i][j]*escalar
        })
        return arr
    }

    static multiply(a,b) {
        var arr = new array2D(a.height, b.width)

        arr.data = arr.map((num, i, j) => {
            let sum = 0
            for (let k = 0; k < a.width; k++) {
                let elm1
                let elm2
                try{
                    if(a.data[i].length == null){
                        elm1 = a.data[i]
                    }else{
                        elm1 = a.data[i][k]
                    }
                    if(b.data[k].length == null){
                        elm2 = b.data[k]
                    }else{
                        elm2 = b.data[k][j]
                    }
                }catch{
                    break
                }
                sum += elm1 * elm2
            }
            return sum
        })

        return arr
    }
}
class neuron{
    constructor(inputNodes,hiddenNodes,outputNodes){
        this.inputNodes = inputNodes
        this.hiddenNodes = hiddenNodes
        this.outputNodes = outputNodes

        this.biasIh = new array2D(hiddenNodes,1)
        this.biasHo = new array2D(outputNodes,1)

        this.weigthsIh = new array2D(hiddenNodes,inputNodes)
        this.weigthsHo = new array2D(outputNodes,hiddenNodes)
    }

    repeatedTrain(data,repeats,lr = 0.1){
        if(!templateFunctions.validateDataSet(data)){
            throw "Invalid DataSet!"
        }
        var allLoss = []
        var loss = 0

        for(var c = 0;c < repeats;c++){
            loss = 0

            for(var r = 0; r < data.inputs.length;r++){
                var response = this.singleTrain(data.inputs[r],data.outputs[r],lr)
                loss += response ** 2
            }

            allLoss.push(loss)
        }

        return {lossArray: allLoss, loss: loss, repeats: repeats, learningRate: lr}
    }
    scoredTrain(data,percentage,lr = 0.1){
        if(!templateFunctions.validateDataSet(data)){
            throw "Invalid DataSet!"
        }
        while(true){
            var err = []

            for(var r = 0; r < data.inputs.length;r++){
                err.push(this.singleTrain(data.inputs[r],data.outputs[r],lr))
            }

            var percentageErr = []
            err.forEach(elm => {
                var a = 0

                elm.forEach(x => {
                    a += Math.abs(100 - templateFunctions.RuleOfThree(.95,100,Math.abs(x[0])))
                })

                percentageErr.push(a / elm.length)
            })

            var averangePercentage = 0

            percentageErr.forEach(elm=>{
                averangePercentage += elm
            })
            averangePercentage /= percentageErr.length

            if(averangePercentage >= percentage){
                break
            }
        }
    }
    singleTrain(inputArr,outputArr,lr = 0.1){
        // Input layer >>> Hidden layer
        var input = new array2D(this.inputNodes,1,inputArr)
        var hidden = array2D.multiply(this.weigthsIh,input)
        hidden = array2D.sum(hidden,this.biasIh)
        hidden.data = hidden.map(templateFunctions.sigmoid)

        // Hidden layer >>> Output layer
        var output = array2D.multiply(this.weigthsHo,hidden)
        output = array2D.sum(output,this.biasHo)
        output.data = output.map(templateFunctions.sigmoid)

        // Output layer >>> Hidden layer
        var expected = new array2D(this.outputNodes,1,outputArr)
        var derivatedOutput = new array2D(this.outputNodes,1,output.map(templateFunctions.dsigmoid))
        var outputError = array2D.sub(expected,output)
        
        var hiddenTransposed = array2D.transpose(hidden)

        var gradient = array2D.hadamard(outputError,derivatedOutput)
        gradient = array2D.escalarMultiply(gradient,lr)

        var weigthsHoDeltas = array2D.multiply(gradient,hiddenTransposed)
        this.weigthsHo = array2D.sum(this.weigthsHo,weigthsHoDeltas)
        
        // Hidden layer >>> Input layer
        var weigthsHoTransposed = array2D.transpose(this.weigthsHo)
        var hiddenError = array2D.multiply(weigthsHoTransposed,outputError)
        var derivatedHidden = new array2D(this.hiddenNodes,1,hidden.map(templateFunctions.dsigmoid))

        var inputTransposed = array2D.transpose(input)

        var gradientHidden = array2D.hadamard(hiddenError,derivatedHidden)
        gradientHidden = array2D.escalarMultiply(gradientHidden, lr)
        
        var weigthsIhDeltas = array2D.multiply(gradientHidden,inputTransposed)
        this.weigthsIh = array2D.sum(this.weigthsIh,weigthsIhDeltas)
    
        // Adjust biasIh
        this.biasHo = array2D.sum(this.biasIh,gradientHidden)

        // Adjust biasHo
        this.biasHo = array2D.sum(this.biasHo,gradient)

        return outputError.toArray()
    }

    execute(inputArr){
        // Input layer >>> Hidden layer
        var input = new array2D(this.inputNodes,1,inputArr)
        var hidden = array2D.multiply(this.weigthsIh,input)
        hidden = array2D.sum(hidden,this.biasIh)
        hidden.data = hidden.map(templateFunctions.sigmoid)

        // Hidden layer >>> Output layer
        var output = array2D.multiply(this.weigthsHo,hidden)
        output = array2D.sum(output,this.biasHo)
        output.data = output.map(templateFunctions.sigmoid)
        
        return array2D.transpose(output).toArray()[0]
    }

    loadRecipe(recipe){
        // Receive a string and convert him to JSON
        var recipe = JSON.parse(recipe)
        this.inputNodes = recipe.inputNodes
        this.hiddenNodes = recipe.hiddenNodes
        this.outputNodes = recipe.outputNodes
        this.biasIh = recipe.biasIh
        this.biasHo = recipe.biasHo
        this.weigthsIh = recipe.weigthsIh
        this.weigthsHo = recipe.weigthsHo
    }

    generateRecipe(){
        // Generate a string
        return JSON.stringify(this)
    }

    static mutate(bot,learningRate){
        // Mutate
        var m_wIh = new array2D(bot.weigthsIh.height,bot.weigthsIh.width)
        var m_wHo = new array2D(bot.weigthsHo.height,bot.weigthsHo.width)
        var m_bIh = new array2D(bot.biasIh.height,bot.biasIh.width)
        var m_bHo = new array2D(bot.biasHo.height,bot.biasHo.width)

        // Mutate the weigths(input to hidden)
        for(var i = 0;i < bot.weigthsIh.height;i++){
            for(var j = 0;j < bot.weigthsIh.width;j++){
                m_wIh.data[i][j] = bot.weigthsIh.data[i][j]+templateFunctions.random(learningRate)
                if(m_wIh.data[i][j] < -1){
                    m_wIh.data[i][j] = -1
                }else if(m_wIh.data[i][j] > 1){
                    m_wIh.data[i][j] = 1
                }
            }
        }

        // Mutate the weigths(hidden to output)
        for(var i = 0;i < bot.weigthsHo.height;i++){
            for(var j = 0;j < bot.weigthsHo.width;j++){
                m_wHo.data[i][j] = bot.weigthsHo.data[i][j]+templateFunctions.random(learningRate)
                if(m_wHo.data[i][j] < -1){
                    m_wHo.data[i][j] = -1
                }else if(m_wHo.data[i][j] > 1){
                    m_wHo.data[i][j] = 1
                }
            }
        }

        // Mutate the bias(input to hidden)
        for(var i = 0;i < bot.biasIh.height;i++){
            for(var j = 0;j < bot.biasIh.width;j++){
                m_bIh.data[i][j] = bot.biasIh.data[i][j]+templateFunctions.random(learningRate)
                if(m_bIh.data[i][j] < -1){
                    m_bIh.data[i][j] = -1
                }else if(m_bIh.data[i][j] > 1){
                    m_bIh.data[i][j] = 1
                }
            }
        }

        // Mutate the bias(hidden to output)
        for(var i = 0;i < bot.biasHo.height;i++){
            for(var j = 0;j < bot.biasHo.width;j++){
                m_bHo.data[i][j] = bot.biasHo.data[i][j]+templateFunctions.random(learningRate)
                if(m_bHo.data[i][j] < -1){
                    m_bHo.data[i][j] = -1
                }else if(m_bHo.data[i][j] > 1){
                    m_bHo.data[i][j] = 1
                }
            }
        }

        // And mutate bot
        var newBot = new neuron(bot.inputNodes, bot.hiddenNodes, bot.outputNodes)
        newBot.weigthsIh = m_wIh
        newBot.weigthsHo = m_wHo
        newBot.biasIh = m_bIh
        newBot.biasHo = m_bHo

        return newBot
    }
}

class templateFunctions{
    static RuleOfThree(a,b,c){
        if(a == 0){
            return 0
        }else{
            return (b * c) / a
        }
    }
    static inverseRuleOfThree(a,b,c){
        if(c == 0){
            return 0
        }else{
            return (a * b) / c
        }
    }
    static sigmoid(x){
        return 1 / (1 + Math.exp(-x))
    }
    static dsigmoid(x){
        return x * (1 - x)
    }
    static random(x){
        return (Math.random()*(x*2))-x
    }
    static objectMap(obj, callback){
        return Object.keys(obj).map(elm => {
            callback(elm, obj[elm])
        })
    }
    static normalizeText(text){
        return text
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
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
