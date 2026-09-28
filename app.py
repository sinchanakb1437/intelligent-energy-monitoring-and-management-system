import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

st.set_page_config(page_title="Intelligent energy monitoring", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#0b1020,#172554,#111827);}
div[data-testid="stMetric"]{
background:rgba(255,255,255,0.08);
padding:15px;border-radius:15px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df=pd.read_csv("household_power_consumption.csv", nrows=15000)
    cols=["Global_active_power","Global_reactive_power","Voltage","Global_intensity"]
    for c in cols:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.dropna()

try:
    df=load_data()
except:
    st.error("Place household_power_consumption.csv beside app.py")
    st.stop()

BILLS_FILE="bill_history.csv"
if not os.path.exists(BILLS_FILE):
    pd.DataFrame(columns=["Month","Units","Amount"]).to_csv(BILLS_FILE,index=False)

st.sidebar.title("⚡ Energy monitoring system")
page=st.sidebar.radio("Navigation",[
"Dashboard","AI Predictor","Billing Center",
"Bill History","Sustainability","Reports","Dataset"
])

avg=df["Global_active_power"].mean()
bill=avg*8*30
saving=bill*0.18
carbon=avg*0.82
health=max(0,min(100,100-avg*5))

if page=="Dashboard":
    st.title("⚡ AI Smart Energy Management")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Energy Score",f"{health:.0f}%")
    c2.metric("Predicted Bill",f"₹{bill:,.0f}")
    c3.metric("Potential Savings",f"₹{saving:,.0f}")
    c4.metric("Carbon Footprint",f"{carbon:.2f} kg")

    st.plotly_chart(
        px.line(df.head(1000),y="Global_active_power",title="Consumption Trend"),
        use_container_width=True
    )

    st.subheader("AI Recommendations")
    st.success("Run heavy appliances during off-peak hours.")
    st.success("Switch to LED lighting.")
    st.success("Reduce standby power consumption.")

elif page=="AI Predictor":
    X=df[["Voltage","Global_intensity"]]
    y=df["Global_active_power"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model=RandomForestRegressor(n_estimators=100,random_state=42)
    model.fit(X_train,y_train)
    score=r2_score(y_test,model.predict(X_test))

    st.metric("Model Accuracy",f"{score*100:.2f}%")
    v=st.number_input("Voltage",230.0)
    i=st.number_input("Current Intensity",15.0)
    if st.button("Predict"):
        st.success(f"Predicted Consumption: {model.predict([[v,i]])[0]:.2f} kW")

elif page=="Billing Center":
    st.title("💰 Smart Billing")
    units=st.number_input("Units Consumed",value=500)
    rate=st.number_input("Rate Per Unit",value=8)
    month=st.text_input("Month","June 2026")

    amount=units*rate
    st.metric("Bill Amount",f"₹{amount:,.2f}")
    st.metric("Saving Opportunity",f"₹{amount*0.15:,.2f}")

    if st.button("Save Bill"):
        bills=pd.read_csv(BILLS_FILE)
        bills.loc[len(bills)] = [month,units,amount]
        bills.to_csv(BILLS_FILE,index=False)
        st.success("Bill saved successfully")

elif page=="Bill History":
    st.title("📜 Previous Electricity Bills")
    bills=pd.read_csv(BILLS_FILE)
    st.dataframe(bills,use_container_width=True)

    if len(bills)>0:
        st.plotly_chart(
            px.bar(bills,x="Month",y="Amount",title="Monthly Bills"),
            use_container_width=True
        )

elif page=="Sustainability":
    st.title("🌱 Sustainability Dashboard")
    st.metric("Carbon Footprint",f"{carbon:.2f} kg")
    st.metric("Efficiency Score",f"{health:.0f}%")
    st.plotly_chart(
        px.pie(values=[health,100-health],names=["Efficient","Waste"]),
        use_container_width=True
    )

elif page=="Reports":
    st.title("📊 Reports")
    st.info(f"Average Power: {avg:.2f} kW")
    st.info(f"Estimated Annual Bill: ₹{bill*12:,.0f}")
    st.info(f"Estimated Annual Savings: ₹{saving*12:,.0f}")

else:
    st.title("Dataset Explorer")
    st.dataframe(df.head(200),use_container_width=True)