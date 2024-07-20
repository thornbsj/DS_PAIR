using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;
using Microsoft.ML.Tokenizers;
using System.Runtime.CompilerServices;
using System.Numerics;


namespace DS_Desc
{
    internal class Generate
    {
        public static Stream token_stream = new FileStream("./tokenizer.json", FileMode.Open);
        public static Bpe tokenizer = new Bpe("./vocab.json", "./merges.txt");
        public static Dictionary<int, char> hf_encoder = new Dictionary<int, char>();
        public static Dictionary<char, int> hf_decoder = new Dictionary<char, int>();
        public static MLContext mlContext = new MLContext();

        private static IReadOnlyList<Token> Encoding_input(string s)
        {
            s = new string(Encoding.UTF8.GetBytes(s).Select(b => hf_encoder[b]).ToArray());
            return tokenizer.Tokenize(s);
        }

        private static string Decoding_output(long[] res_token)
        {
            string decoded = Bpe.Decoder.Decode(res_token.Select(id => tokenizer.IdToToken((int)id)!));
            decoded = Encoding.UTF8.GetString(decoded.Select(c => (byte)hf_decoder[c]).ToArray());
            return decoded;
        }

        public class OnnxInput
        {
            [VectorType]
            [ColumnName("input_ids")]
            public long[] InputIds { get; set; }

            [VectorType]
            [ColumnName("attention_mask")]
            public long[] AttentionMask { get; set; }
        }

        public class OnnxOutput
        {
            [VectorType]
            [ColumnName("logits")]
            public float[] logits { get; set; }
        }
        static ITransformer GetPredictionPipeline(MLContext mlContext)
        {
            var inputColumns = new string[]{
                "input_ids", "attention_mask"
            };


            var outputColumns = new string[] { "logits" };
            var onnxPredictionPipeline =mlContext.Transforms.ApplyOnnxModel(
                outputColumnNames:outputColumns,
                inputColumnNames:inputColumns,
                modelFile: "./DS_Desc_ONNX_Model/model.onnx"
                );
            var emptyDv = mlContext.Data.LoadFromEnumerable(new OnnxInput[] { });
            return onnxPredictionPipeline.Fit(emptyDv);
        }

        public static string Generate_result(string s)
        {
            var onnxPredictionPipeline = GetPredictionPipeline(mlContext);
            var onnxPredictionEngine = mlContext.Model.CreatePredictionEngine<OnnxInput, OnnxOutput>(onnxPredictionPipeline);

            s = "被人们冠以“正义”的剑，能够驱散黑暗并治疗自身";
            var tokenizerEncodedResult = Encoding_input(s);
            long[] InputIds = tokenizerEncodedResult.Select(x => (long)x.Id).ToArray();//new long[tokenizerEncodedResult.Count];
            Array.Resize(ref InputIds,1000);
            long[] attention_masks = Enumerable.Repeat(1L, tokenizerEncodedResult.Count).ToArray();
            Array.Resize(ref attention_masks, 1000);
            var ipt = new OnnxInput
            {
                InputIds = InputIds,
                AttentionMask = attention_masks
            };

            float[] res = onnxPredictionEngine.Predict(ipt).logits;

            long[]res_token = new long[res.Length/ 250880];
            int res_token_idx = 0;
            for (int i = 0; i < res.Length; i+= 250880) {
                int maxIndexInGroup = -1;
                float maxValueInGroup = float.MinValue;
                for (int j = i; j < Math.Min(i + 250880, res.Length); j++)
                {
                    // 如果当前元素大于组中的最大值  
                    if (res[j] > maxValueInGroup)
                    {
                        maxValueInGroup = res[j]; // 更新最大值  
                        maxIndexInGroup = j - i; // 更新最大值在组中的下标（相对于组的起始位置）  
                    }
                }
                res_token[res_token_idx] = maxIndexInGroup;
                res_token_idx ++;
            }
            s = Decoding_output(res_token);
            return s + "initial";
        }
    }
}
