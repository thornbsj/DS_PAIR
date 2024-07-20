namespace DS_Desc
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }


        private void button1_Click_1(object sender, EventArgs e)
        {
            string ipt = this.foreverTextBox1.Text;
            string output = Generate.Generate_result(ipt);
            this.foreverTextBox2.Text = output;
        }

        private void foreverButton1_Click(object sender, EventArgs e)
        {
            this.foreverButton1.Text = "生成中，请耐心等待";
            string ipt = this.foreverTextBox1.Text;
            string output = Generate.Generate_result(ipt);
            this.foreverTextBox2.Text = output.Replace("\n", Environment.NewLine);
            this.foreverButton1.Text = "生成描述";
        }
    }
}
