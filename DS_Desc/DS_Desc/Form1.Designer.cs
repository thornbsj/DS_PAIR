namespace DS_Desc
{
    partial class Form1
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            foreverTextBox1 = new ReaLTaiizor.Controls.ForeverTextBox();
            foreverTextBox2 = new ReaLTaiizor.Controls.ForeverTextBox();
            foreverButton1 = new ReaLTaiizor.Controls.ForeverButton();
            SuspendLayout();
            // 
            // foreverTextBox1
            // 
            foreverTextBox1.BackColor = Color.Transparent;
            foreverTextBox1.BaseColor = Color.FromArgb(45, 47, 49);
            foreverTextBox1.BorderColor = Color.Orange;
            foreverTextBox1.FocusOnHover = false;
            foreverTextBox1.Font = new Font("仿宋", 12F);
            foreverTextBox1.ForeColor = Color.FromArgb(192, 192, 192);
            foreverTextBox1.Location = new Point(499, 97);
            foreverTextBox1.MaxLength = 32767;
            foreverTextBox1.Multiline = true;
            foreverTextBox1.Name = "foreverTextBox1";
            foreverTextBox1.ReadOnly = false;
            foreverTextBox1.Size = new Size(566, 110);
            foreverTextBox1.TabIndex = 0;
            foreverTextBox1.Text = "输入前半段描述";
            foreverTextBox1.TextAlign = HorizontalAlignment.Center;
            foreverTextBox1.UseSystemPasswordChar = false;
            // 
            // foreverTextBox2
            // 
            foreverTextBox2.BackColor = Color.Transparent;
            foreverTextBox2.BaseColor = Color.FromArgb(45, 47, 49);
            foreverTextBox2.BorderColor = Color.Orange;
            foreverTextBox2.FocusOnHover = false;
            foreverTextBox2.Font = new Font("仿宋", 12F, FontStyle.Underline);
            foreverTextBox2.ForeColor = Color.FromArgb(192, 192, 192);
            foreverTextBox2.Location = new Point(508, 448);
            foreverTextBox2.MaxLength = 32767;
            foreverTextBox2.Multiline = true;
            foreverTextBox2.Name = "foreverTextBox2";
            foreverTextBox2.ReadOnly = true;
            foreverTextBox2.Size = new Size(566, 321);
            foreverTextBox2.TabIndex = 1;
            foreverTextBox2.TextAlign = HorizontalAlignment.Center;
            foreverTextBox2.UseSystemPasswordChar = false;
            // 
            // foreverButton1
            // 
            foreverButton1.BackColor = Color.Transparent;
            foreverButton1.BaseColor = Color.FromArgb(45, 47, 49);
            foreverButton1.Font = new Font("仿宋", 12F);
            foreverButton1.Location = new Point(625, 262);
            foreverButton1.Name = "foreverButton1";
            foreverButton1.Rounded = false;
            foreverButton1.Size = new Size(296, 71);
            foreverButton1.TabIndex = 2;
            foreverButton1.Text = "生成描述";
            foreverButton1.TextColor = Color.FromArgb(243, 243, 243);
            foreverButton1.Click += foreverButton1_Click;
            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(11F, 24F);
            AutoScaleMode = AutoScaleMode.Font;
            AutoSize = true;
            BackgroundImage = Properties.Resources.cover;
            BackgroundImageLayout = ImageLayout.Stretch;
            ClientSize = new Size(1532, 901);
            Controls.Add(foreverButton1);
            Controls.Add(foreverTextBox2);
            Controls.Add(foreverTextBox1);
            FormBorderStyle = FormBorderStyle.FixedSingle;
            Icon = (Icon)resources.GetObject("$this.Icon");
            MaximizeBox = false;
            Name = "Form1";
            Text = "黑暗之魂物品描述生成";
            ResumeLayout(false);
        }

        #endregion

        private ReaLTaiizor.Controls.ForeverTextBox foreverTextBox1;
        private ReaLTaiizor.Controls.ForeverTextBox foreverTextBox2;
        private ReaLTaiizor.Controls.ForeverButton foreverButton1;
    }
}
