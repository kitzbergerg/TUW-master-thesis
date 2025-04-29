import * as ort from 'onnxruntime-web';



async function createSession(modelPath, preferredOutputLocation, executionProviders = "webnn") {
    return await ort.InferenceSession.create(modelPath, {
        executionProviders: [executionProviders],
        preferredOutputLocation: preferredOutputLocation
    });
}

function arrayEquals(a, b) {
    return a.length === b.length && a.every((val, index) => val === b[index]);
}

async function testModel() {
    const dims = [1, 3, 224, 224];
    const data = Float32Array.from(Array(dims.reduce((acc, el) => acc * el)).fill(0));
    const tensor = new ort.Tensor('float32', data, dims);
    const feeds = { data: tensor };

    // Test full model
    const session = await createSession('model/squeezenet1.1-7.onnx', 'cpu', 'webnn');
    const out = await session.run(feeds);
    console.log(out);

    // Test split model
    const session_p1 = await createSession('model/squeezenet1.1-7_p1.onnx', 'ml-tensor', 'webnn');
    const out_p1 = await session_p1.run(feeds);
    console.log(out_p1);

    const session_p2 = await createSession('model/squeezenet1.1-7_p2.onnx', 'cpu', 'webnn');
    const out_p2 = await session_p2.run(out_p1);
    console.log(out_p2);

    // assert
    console.assert(arrayEquals(out.squeezenet0_flatten0_reshape0.cpuData, out_p2.squeezenet0_flatten0_reshape0.cpuData));
    console.log("Split squeezenet gives same result as full variant")
}
window.testModel = testModel
