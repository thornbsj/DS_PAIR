using System.Linq;

namespace DS_Desc
{
    internal static class Program
    {
        /// <summary>
        ///  The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            // To customize application configuration such as set high DPI settings or default font,
            // see https://aka.ms/applicationconfiguration.
            for (int c = '!'; c <= '~'; c++) Generate.hf_encoder.Add(c, (char)c);
            for (int c = '¡'; c <= '¬'; c++) Generate.hf_encoder.Add(c, (char)c);
            for (int c = '®'; c <= 'ÿ'; c++) Generate.hf_encoder.Add(c, (char)c);
            int n = 0;
            for (int c = 0; c < 256; c++)
            {
                if (!Generate.hf_encoder.ContainsKey(c))
                    Generate.hf_encoder.Add(c, (char)(256 + n++));
            }

            Generate.hf_decoder = Generate.hf_encoder.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
            Generate.Initial_Model();
            ApplicationConfiguration.Initialize();
            Application.Run(new Form1());
        }
    }
}