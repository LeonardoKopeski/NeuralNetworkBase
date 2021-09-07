# NeuralNetworkLibrary
## Idea
Basicamente, eu estava entediado, e resolvi criar a base de uma IA, sem usar nenhuma biblioteca

## Docs

Para criar um neuronio, devemos declarar um novo objeto da classe neuron
```
var n = new neuron(<InputNodes>,<HiddenNodes>,<OutputNodes>)
```

Para treinar o neuronio, podemos usar dois modelos, mas para ambos os dois, precimaos criar um dataset
```
var dataSet = {
  inputs:[
    <input1>,
    <input2>,
    <input3>,
    <input4>
  ],
  outputs:[
    <outputReferenteAoInput1>,
    <outputReferenteAoInput2>,
    <outputReferenteAoInput3>,
    <outputReferenteAoInput4>
  ]
}
```

Para treinar de forma repetida, usamos:
```
n.repeatedTrain(<dataset>, <repetições>,<learningRate>)
```

Para treinar até ele atingir um determinado score, usamos:
```
n.scoredTrain(<dataset>, <score>,<learningRate>)
```

Para testar o nosso neuronio, devemos usar:
```
n.execute(<input>)
```

Caso queira exportar o neuronio, use:
```
n.generateRecipe()
```

Mas, caso queira importar...
```
n.loadRecipe(<recipe>)
```

(Desatualizado)
Caso queira gerar uma mutação do neuronio, use:
```
neuron.mutate(n, <learningRate>)
```

## Example
Vou usar o exemplo de um XOR(ou exclusivo):
```
var n = new neuron(2,3,1)
var dataSet = {
  inputs:[
    [1,1],
    [0,0],
    [0,1],
    [1,0]
  ],
  outputs:[
    [0],
    [0],
    [1],
    [1]
  ]
}
n.repeatedTrain(dataSet, 5000,.3)

console.log("XOR de 0 e 0 resulta em: "+Math.round(n.execute([0,0])[0]))
console.log("XOR de 1 e 0 resulta em: "+Math.round(n.execute([1,0])[0]))
console.log("XOR de 0 e 1 resulta em: "+Math.round(n.execute([0,1])[0]))
console.log("XOR de 1 e 1 resulta em: "+Math.round(n.execute([1,1])[0]))
/*
XOR de 0 e 0 resulta em: 0
XOR de 1 e 0 resulta em: 1
XOR de 0 e 1 resulta em: 1
XOR de 1 e 1 resulta em: 0
*/
```

## Credits
- Author: Leonardo Kopeski
- Inspirated By: José Bezerra

## License
MIT
