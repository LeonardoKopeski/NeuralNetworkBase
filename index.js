var nnl = require("./library.js")

var n = new nnl.neuralNetwork([2,3,2,1])

console.log(n.execute([0,0]))

n.train([0,0], [0], .3)

console.log(n.execute([0,0]))