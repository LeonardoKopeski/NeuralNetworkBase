# NeuralNetworkLibrary
## Idea
Basicamente, eu estava entediado, e resolvi criar a base de uma IA, sem usar nenhuma biblioteca

## Docs

Para criar um neuronio, devemos declarar um novo objeto da classe neuron
```
var n = new neuron([<nodes>])
```

Para treinar o neuronio, podemos usar dois modelos, mas para ambos os dois, precimaos criar um dataset
```
n.set(input, expectedOutput)
```

Para treinar de forma repetida, usamos:
```
n.train(<repetições>,<learningRate>)
```

Para testar o nosso neuronio, devemos usar:
```
n.execute(<input>)
// Ou
n.asyncExecute(<input>)
```

Caso queira exportar o neuronio, use:
```
n.generateRecipe()
```

Mas, caso queira importar...
```
var n = neuron.loadRecipe(<recipe>)
```

## Example
Vou usar o exemplo de um XOR(ou exclusivo):
```
var n = new neuron([2,3,1])

n.learningRate = .3

n.set([0, 0], [0])
n.set([0, 1], [1])
n.set([1, 0], [1])
n.set([1, 1], [0])

n.train(5000).then(()=>{
  console.log("XOR de 0 e 0 resulta em: "+Math.round(n.execute([0,0])[0]))
  console.log("XOR de 1 e 0 resulta em: "+Math.round(n.execute([1,0])[0]))
  console.log("XOR de 0 e 1 resulta em: "+Math.round(n.execute([0,1])[0]))
  console.log("XOR de 1 e 1 resulta em: "+Math.round(n.execute([1,1])[0]))
})

/*
XOR de 0 e 0 resulta em: 0
XOR de 1 e 0 resulta em: 1
XOR de 0 e 1 resulta em: 1
XOR de 1 e 1 resulta em: 0
*/
```

## Credits
- Author: Leonardo Kopeski

## License
MIT
