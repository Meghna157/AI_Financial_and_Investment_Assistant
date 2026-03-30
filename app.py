import os
import requests
import base64
import io
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from dotenv import load_dotenv
from PIL import Image
from fpdf import FPDF

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="AI Financial Analysis Pro", layout="wide")

# =====================================================
# 🎨 PROFESSIONAL UI
# =====================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 18px;
}
h1 {
    font-size: 42px !important;
    font-weight: 700 !important;
    color: #38bdf8 !important;
}
h2, h3 {
    font-size: 26px !important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #020617);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="metric-container"] {
    background: #0f172a;
    border-radius: 14px;
    padding: 15px;
}
button[kind="primary"] {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
}
</style>
""", unsafe_allow_html=True)

st.title("📊 AI Financial Analysis Assistant")
st.caption("Multimodal RAG + Vision + Live Sentiment")
st.warning("Educational purpose only. Not investment advice.")

load_dotenv()

# =====================================================
# LLM
# =====================================================
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
vision_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1)

# =====================================================
# CACHING
# =====================================================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def build_vectorstore(path):
    if not os.path.exists(path): return None
    embeddings = load_embeddings()
    with open(path, "r", encoding="utf-8") as f:
        report_text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(report_text)
    return FAISS.from_texts(chunks, embeddings)

@st.cache_data(ttl=300)
def load_stock_data(symbol):
    stock = yf.Ticker(symbol)
    return stock.info, stock.history(period="1y")

@st.cache_data(ttl=600)
def fetch_news(url):
    return requests.get(url).json()

@st.cache_data(ttl=1800)
def get_sentiment(headline):
    return llm.invoke(f"Sentiment for: {headline}").content

# =====================================================
# HELPERS
# =====================================================
def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

def format_number(num):
    if num is None: return "N/A"
    if abs(num) > 1_000_000_000: return f"{num/1_000_000_000:.2f} B"
    if abs(num) > 1_000_000: return f"{num/1_000_000:.2f} M"
    return f"{num:,.2f}"

#def show_beginner_definitions():
    with st.expander("🎓 Learn Metrics"):
        st.markdown("""
        * Market Cap  
        * P/E Ratio  
        * Profit Margin  
        * ROE  """)
    

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.markdown("<h2 style='color:white;'>🏢 Select Company</h2>", unsafe_allow_html=True)

stock_options = {"Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL", "Infosys": "INFY", "TCS": "TCS.NS"}

dropdown_selection = st.sidebar.selectbox("", ["-- Select --"] + list(stock_options.keys()))

# =====================================================
# MAIN
# =====================================================
if dropdown_selection != "-- Select --":

    symbol = stock_options[dropdown_selection]
    selected_stock = dropdown_selection

    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "summary_text" not in st.session_state: st.session_state.summary_text = ""
    if "latest_vision_analysis" not in st.session_state: st.session_state.latest_vision_analysis = None
    if "retriever" not in st.session_state: st.session_state.retriever = None

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Market Data", "⚖️ Comparison", "📄 AI Report", "🗞️ News"])

    # ================= TAB 1 =================
    with tab1:
        #show_beginner_definitions()

        with st.spinner("Loading market data..."):
            info, hist = load_stock_data(symbol)

        st.subheader(f"{selected_stock} ({symbol})")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            label="Market Cap 🧾",
            value=format_number(info.get("marketCap")),
            help="Total market value of all outstanding shares. Indicates company size."
        )

        c2.metric(
            label="P/E Ratio 📊",
            value=info.get("trailingPE", "N/A"),
            help="Price-to-Earnings ratio: how much investors pay for $1 of earnings."
        )

        c3.metric(
            label="Profit Margin 💰",
            value=f"{info.get('profitMargins',0)*100:.2f}%" if info.get("profitMargins") else "N/A",
            help="Percentage of revenue that becomes profit after expenses."
        )

        c4.metric(
        label="ROE ⚖️",
        value=f"{info.get('returnOnEquity',0)*100:.2f}%" if info.get("returnOnEquity") else "N/A",
        help="Return on Equity: efficiency of using shareholder funds."
        )
        
        
        
        #c1.metric("Market Cap", format_number(info.get("marketCap")))
        #c2.metric("P/E Ratio", info.get("trailingPE", "N/A"))
        #c3.metric("Profit Margin", f"{info.get('profitMargins',0)*100:.2f}%")
        #c4.metric("ROE", f"{info.get('returnOnEquity',0)*100:.2f}%")

        fig = go.Figure(data=[go.Scatter(x=hist.index, y=hist['Close'])])
        st.plotly_chart(fig, use_container_width=True)

    # ================= TAB 2 =================
    with tab2:
        compare_list = st.multiselect("Peers:", list(stock_options.keys()), default=[selected_stock])

        fig = go.Figure()
        performance_data = {}

        for name in compare_list:
            _, c_data = load_stock_data(stock_options[name])

            if not c_data.empty:
                norm = (c_data["Close"] / c_data["Close"].iloc[0]) * 100
                fig.add_trace(go.Scatter(x=c_data.index, y=norm, name=name))

                perf = ((c_data["Close"].iloc[-1] - c_data["Close"].iloc[0]) / c_data["Close"].iloc[0]) * 100
                performance_data[name] = perf

        st.plotly_chart(fig, use_container_width=True)

        if performance_data:
            st.subheader("📊 Performance Summary")

            for comp, val in performance_data.items():
                st.write(f"**{comp}**: {val:.2f}% return")

            best_stock = max(performance_data, key=performance_data.get)
            st.success(f"🏆 Best Performer: {best_stock}")

            if st.button("🧠 AI Insight"):
                summary_text = "\n".join([f"{k}: {v:.2f}%" for k, v in performance_data.items()])
                insight = llm.invoke(
    f"""
    You are a financial analyst.

    The following data represents 1-year stock returns (NOT market share):

    {summary_text}

    Tasks:
    - Identify the best and worst performing stocks
    - Compare their performance
    - Provide a short investor-friendly insight
    - Do NOT assume these are market share percentages

    Keep the explanation clear and professional.
    """
).content
                st.info(insight)

    # ================= TAB 3 =================
    with tab3:
        clean_name = symbol.split('.')[0]
        doc_path = Path(__file__).parent / "docs" / "txt" / f"{clean_name}_annual_report.txt"

        if doc_path.exists():

            if st.button("🚀 Load AI Report Analysis"):
                with st.spinner("Initializing AI engine..."):
                    vs = build_vectorstore(str(doc_path))
                    st.session_state.retriever = vs.as_retriever(search_kwargs={"k": 3})

            if st.session_state.retriever:
                retriever = st.session_state.retriever

                # SUMMARY
                st.subheader("📑 Executive Summary")
                if not st.session_state.summary_text:
                    context = "\n".join([d.page_content for d in retriever.invoke("Key highlights")])
                    st.session_state.summary_text = llm.invoke(context).content
                st.info(st.session_state.summary_text)

                # VISION
                st.subheader("📊 Chart Interpretation")
                chart_img = st.file_uploader("Upload chart:", type=["png","jpg"])

                if chart_img:
                    st.image(chart_img)
                    if st.button("Analyze Chart"):
                        b64 = encode_image(chart_img)
                        msg = HumanMessage(content=[
                            {"type":"text","text":"Explain this chart"},
                            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
                        ])
                        st.session_state.latest_vision_analysis = vision_llm.invoke([msg]).content

                if st.session_state.latest_vision_analysis:
                    st.write(st.session_state.latest_vision_analysis)

                # CHAT
                st.subheader("💬 Ask")
                for m in st.session_state.chat_history:
                    st.write(m["content"])

                q = st.text_input("Ask question")
                if q:
                    context = "\n".join([d.page_content for d in retriever.invoke(q)])
                    ans = llm.invoke(context).content
                    st.write(ans)
            
    # ================= PDF EXPORT WITH PREVIEW =================

                has_content = (
                    st.session_state.summary_text or
                    st.session_state.latest_vision_analysis or
                    st.session_state.chat_history
                )

                if has_content:
                    st.divider()
                    st.subheader("📄 Export AI Report")

                    if st.button("📄 Generate Report Preview"):
                        with st.spinner("Generating preview..."):

                            pdf = FPDF()
                            pdf.add_page()

                            pdf.set_font("Arial", 'B', 16)
                            pdf.cell(200, 10, f"Analysis Report: {selected_stock}", ln=True, align='C')

                            pdf.ln(10)
                            pdf.set_font("Arial", '', 11)

                            # Summary
                            if st.session_state.summary_text:
                                pdf.multi_cell(0, 8, txt=f"EXECUTIVE SUMMARY:\n{st.session_state.summary_text}")

                            # Chart analysis
                            if st.session_state.latest_vision_analysis:
                                pdf.multi_cell(0, 8, txt=f"\nCHART ANALYSIS:\n{st.session_state.latest_vision_analysis}")

                            # Chat
                            if st.session_state.chat_history:
                                pdf.multi_cell(0, 8, txt="\nDETAILED Q&A:\n")

                                for m in st.session_state.chat_history:
                                    role = m["role"].upper()
                                    content = m["content"]

                                    pdf.multi_cell(
                                    0, 8,
                                    txt=f"{role}: {content}".encode('latin-1', 'replace').decode('latin-1')
                                    )

                            # Generate PDF bytes
                            pdf_bytes = pdf.output(dest='S').encode('latin-1')
                            # Save in session for download
                            st.session_state["pdf_bytes"] = pdf_bytes

                            # ================= PREVIEW =================
                            import base64
                            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

                            pdf_display = f'''
                            <iframe 
                                src="data:application/pdf;base64,{base64_pdf}" 
                                width="100%" 
                                height="600px" 
                                type="application/pdf">
                            </iframe>
                            '''

                            st.markdown("### 👀 Preview")
                            st.markdown(pdf_display, unsafe_allow_html=True)

                    # ================= DOWNLOAD BUTTON =================
                    if "pdf_bytes" in st.session_state:
                        st.download_button(
                            "📥 Download PDF",
                            data=st.session_state["pdf_bytes"],
                            file_name=f"{clean_name}_AI_Report.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.info("ℹ️ Generate insights to enable PDF export.")

    # ================= TAB 4 =================
    with tab4:
        key = os.getenv("FINNHUB_API_KEY", "").strip()
        if key:
            from datetime import datetime, timedelta

            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')

            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={key}"
            #url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&token={key}"
            res = fetch_news(url)

# ✅ Check if valid list
            if isinstance(res, list):
                for art in res[:5]:
                    with st.expander(art.get('headline', 'No Title')):
                        sentiment = get_sentiment(art.get('headline', ''))
                        st.info(sentiment)
                        st.link_button("Source", art.get('url', '#'))
            else:
                    st.warning("⚠️ Unable to fetch news. Check API key or try again later.")

else:
    st.info("👈 Select a stock")