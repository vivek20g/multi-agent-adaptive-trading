from agents import Agent, ModelSettings, Runner
from tools import get_stock_technical_signals,get_lstm_breakout_signal

class TechnicalAnalysisAgent:
    def __init__(self):
        SAMPLE_PROMPT = (
          "You are a technical analysis agent. Given a stockname, you must use get_stock_technical_signals to retrieve "
          "Bollinger Bands, RSI, MACD, EMA, and ADX. Given the stockname, you must also use get_lstm_breakout_signal function to get LSTM breakout signal along with LSTM model classification report. "
          "The LSTM model provides Long Buy, Short Sell or No Action signal with confidence score. "
          "Analyze these indicators & LSTM breakout signal and recommend buy or sell with reasoning. "
          "Mention in your summary, both LSTM breakout prediction as well as technical indicators analysis. "
        )
        self.agent = Agent(
                  name="TechnicalAnalysisAgent",
                  instructions=SAMPLE_PROMPT,
                  tools=[get_lstm_breakout_signal, get_stock_technical_signals],
                  model_settings=ModelSettings(tool_choice="required")
                )
    async def run(self, query:str):
        result = await Runner.run(self.agent, input=query)
        return result