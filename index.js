const ACTIVATION = {
    SIGMOID: x => 1 / (1 + Math.pow(Math.E, -x)),
    RELU: x => Math.max(0, x),
    IDENTITY: x => x
}

const DERIVATIVES = {
    SIGMOID: x => x * (1 - x),
    RELU: x => Number(x > 0),
    IDENTITY: x => 1
}

const createWeights = (fromSize, toSize) => {
    return Array(toSize).fill(Array(fromSize).fill(null))
        .map(list => list.map(() => Math.random()))
}

const matWithValue = (x, y, value) => {
    return Array(x).fill(Array(y).fill(value))
}

const realDot = (m1, m2) => {
    return m1.map((v1, i) => v1 * m2[i])
            .reduce((sum, x) => sum + x)
}

const dot = (m1, m2) => {
    return m1.map(v1 => m2.map(v2 => realDot(v1, v2)))
}

const asyncRealDot = (m1, m2) => {
    return m1.map((v1, i) => v1 * m2[i])
            .reduce((sum, x) => sum + x)
}

const asyncDot = (m1, m2) => {
    return Promise.all(m1.map(v1 => {
        return new Promise(async resolve => {
            resolve(await Promise.all(m2.map(v2 => {
                return new Promise(async resolve => {
                    resolve( await asyncRealDot(v1, v2) )
                })
            })))
        })
    }))
}

const realMinus = (m1, m2) => {
    return m1.map((v1, i) => v1 - m2[i])
}

const minus = (m1, m2) => {
    return m1.map((v1, i) => realMinus(v1, m2[i]))
}

const repeatArray = (m, times) => {
    for (let i = 0; i < times; i++) {
        m = m.concat(m)
    }

    return m
}

const realMult = (m1, m2) => {
    const diff = Math.abs(m1.length - m2.length)

    if (m1.length > m2.length) {
        const times = Math.floor(diff / m2.length)
        const extra = diff % m2.length
        m2 = repeatArray(m2, times).concat(m2.slice(0, extra))
    } else if (m1.length < m2.length) {
        const times = Math.floor(diff / m1.length)
        const extra = diff % m1.length
        m1 = repeatArray(m1, times).concat(m1.slice(0, extra))
    }

    return m1.map((v1, i) => v1 * m2[i])
}

const mult = (m1, m2) => {
    return m1.map((v1, i) => realMult(v1, m2[i]))
}

const activate = (products, fn) => {
    return products.map(product => {
        return product.map(value => ACTIVATION[fn](value))
    })
}

const derive = (errors, fn) => {
    return errors.map(error => {
        return error.map(value => DERIVATIVES[fn](value))
    })
}

const transpose = m => {
    return Array(m[0].length).fill(Array(m.length).fill(null))
        .map((n, i) => {
            return n.map((v, j) => {
                return m[j][i]
            })
        })
}

(async () => {
    console.time('Run time')
    
    let input = [[0, 1], [1, 0]]
    const expectedOutput = [[1], [1]]
    const learningRate = 0.001

    let weights1 = createWeights(2, 3)
    let weights2 = createWeights(3, 1)

    // Feedforward

    let product1 = dot(input, weights1)
    let activated1 = activate(product1, 'SIGMOID')

    let product2 = dot(activated1, weights2)
    let output = activate(product2, 'SIGMOID')
    
    console.log("Output:", output)

    // Backprop

    let outputError = minus(expectedOutput, output)
    let outputDelta = mult(outputError, derive(output, 'SIGMOID'))

    let product2Error = dot(outputDelta, transpose(weights2))
    let product2Delta = mult(product2Error, derive(product2, 'SIGMOID'))

    // Update
    

    console.timeEnd('Run time')
})()