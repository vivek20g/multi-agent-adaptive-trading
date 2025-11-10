from agents import Agent, ModelSettings, Runner

class TradeExecutionAgent:
    def __init__(self):
        SAMPLE_PROMPT = (
          """
          You are a text parser assistant responsible for extracting trade recommendations from previously run portfolio management agent.
            If you find a BUY or SELL trade recommendation, you must extract the details of the trade and respond with the following **strict JSON format**:
            {
              "action": "<BUY/SELL>",
              "symbol": "<TICKER>",
              "quantity": <NUMBER>,
              "entry": "market" | "limit",
              "stop_loss": <PRICE>,
              "take_profit": <PRICE> (optional),
              "rationale": "<brief explanation of the trade>"
            }
            The quanity is always a Number/Integer, so deduce the number of stocks from the executive summary always.
            If you find that any trade is not recommended or the prior signals have been rejected by the portfolio maangement agent, respond with:
            {
              "action": "HOLD",
              "rationale": "<brief explanation of why no trade is recommended>"
            }
            Do not include any other text or formatting outside the JSON structure.
            """
        )
        self.agent = Agent(
                  name="TradeExecutionAgent",
                  instructions=SAMPLE_PROMPT
                )
    async def run(self, query:str):
        result = await Runner.run(self.agent, input=query)
        return result