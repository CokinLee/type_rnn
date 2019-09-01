package main

import (
	"fmt"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// CharacterTable 字符表
type CharacterTable struct {
}

func main() {
	// 句子最大长度
	const MAXLEN int = 20
	// 将文本转换为id序列，为了实验方便直接使用转换好的ID序列即可
	inputData := [1][MAXLEN]float32{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 208.0, 659.0, 180.0, 408.0, 42.0, 547.0, 829.0, 285.0, 334.0, 42.0, 642.0, 81.0, 800.0}}
	tensor, err := tf.NewTensor(inputData)
	if err != nil {
		fmt.Printf("Error NewTensor: err: %s", err.Error())
		return
	}
	//读取模型
	model, err := tf.LoadSavedModel("cnnModel", []string{"myTag"}, nil)
	if err != nil {
		fmt.Printf("Error loading Saved Model: %s\n", err.Error())
		return
	}
	// 识别
	result, err := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			// python版tensorflow/keras中定义的输入层input_layer
			model.Graph.Operation("input_layer").Output(0): tensor,
		},
		[]tf.Output{
			// python版tensorflow/keras中定义的输出层output_layer
			model.Graph.Operation("output_layer/Softmax").Output(0),
		},
		nil,
	)

	if err != nil {
		fmt.Printf("Error running the session with input, err: %s  ", err.Error())
		return
	}
	// 输出结果，interface{}格式
	fmt.Printf("Result value: %v", result[0].Value())
}
