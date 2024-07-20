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
using System.Text.RegularExpressions;


namespace DS_Desc
{
    internal class Generate
    {
        public static Stream token_stream = new FileStream("./tokenizer.json", FileMode.Open);
        public static Bpe tokenizer = new Bpe("./vocab.json", "./merges.txt");
        public static Dictionary<int, char> hf_encoder = new Dictionary<int, char>();
        public static Dictionary<char, int> hf_decoder = new Dictionary<char, int>();
        public static MLContext mlContext = new MLContext();
        public static PredictionEngine<OnnxInput, OnnxOutput>onnxPredictionEngine = null;

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
        static long GenNextToken(PredictionEngine<OnnxInput,OnnxOutput> onnxPredictionEngine, List<long> InputIds,List<long> attention_masks){
            int token_size = 250880;
            var ipt = new OnnxInput
            {
                InputIds = InputIds.ToArray(),
                AttentionMask = attention_masks.ToArray()
            };

            float[] res_all = onnxPredictionEngine.Predict(ipt).logits;
            float[] res = res_all.Skip(res_all.Length - token_size).ToArray();
            
            int res_token = -1;
            float maxValueInGroup = float.MinValue;
            for (int j = 0; j < res.Length; j++)
            {
                if (res[j] > maxValueInGroup)
                {
                    maxValueInGroup = res[j];
                    res_token = j;
                }
            }
            return res_token;
        }
        public static void Initial_Model()
        {
            var onnxPredictionPipeline = GetPredictionPipeline(mlContext);
            onnxPredictionEngine = mlContext.Model.CreatePredictionEngine<OnnxInput, OnnxOutput>(onnxPredictionPipeline);
        }
        public static string Generate_result(string s)
        {
            var tokenizerEncodedResult = Encoding_input(s);
            List<long> InputIds = new List<long>(tokenizerEncodedResult.Select(x => (long)x.Id));
            List<long> attention_masks = new List<long>(Enumerable.Repeat(1L, tokenizerEncodedResult.Count));
            long res_token=-1;
            int EOS_TOKEN = 2;
            int MAX_LENGTH = 256;
            while (res_token != EOS_TOKEN && InputIds.Count<MAX_LENGTH){
                res_token = GenNextToken(onnxPredictionEngine, InputIds, attention_masks);
                InputIds.Add(res_token);
                attention_masks.Add(1);
            }
            
            s = Decoding_output(InputIds.Take(InputIds.Count-1).ToArray());
            HashSet<string> wrap_chrs = new HashSet<string>();
            wrap_chrs.Add("……");
            wrap_chrs.Add("——");
            HashSet<char> ignore_chr = new HashSet<char>{ '\"', '“', '”', '…', '—' };
            for (int i=0;i<s.Length;i++) {
                char c= s[i];
                if(!Regex.IsMatch(c.ToString(), @"^[\u4e00-\u9fff]+$") && !ignore_chr.Contains(c))
                {
                    wrap_chrs.Add(c.ToString());
                }
            }
            foreach (string c in wrap_chrs) {
                s = s.Replace(c,c+"\n");
            }
            return s;
        }
    }
}
