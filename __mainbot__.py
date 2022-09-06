from apscheduler.schedulers.asyncio import AsyncIOScheduler
from OnlineBot.MACD import bot as MACDBOT
from OnlineBot.StochAstic import bot as StochAsticBOT
from OnlineBot.RSI import bot as RSIBOT
import threading
import schedule
import time

try:
    import asyncio
except ImportError:
    import trollius as asyncio

scheduler_trader_macd_div = AsyncIOScheduler()
scheduler_trader_stochastic_div = AsyncIOScheduler()
scheduler_trader_rsi_div = AsyncIOScheduler()
scheduler_news = AsyncIOScheduler()

account_name = 'mehrshadpc'
symbol = 'XAUUSD_i'


def trader_macd_div_threaded():
    job_thread = threading.Thread(target=MACDBOT.Run, args = [symbol, account_name])
    job_thread.start()
    job_thread.join()

def trader_stochastic_div_threaded():
    job_thread = threading.Thread(target=StochAsticBOT.Run, args = [symbol, account_name])
    job_thread.start()
    job_thread.join()

def trader_rsi_div_threaded():
    job_thread = threading.Thread(target=RSIBOT.Run, args = [symbol, account_name])
    job_thread.start()
    job_thread.join()

def news_threaded():
	job_thread = threading.Thread(target=news_task)
	job_thread.start()
	job_thread.join()

def Run():
	# Schedules the job_function to be executed Monday through Friday at between 12-16 at specific times. 
	minute_trader = '0,5,10,15,20,25,30,35,40,45,50,55'
	days = 'sat,sun,mon,tue,wed,thu,fri'

	minute_news = '10'
	hour_news = '00,12'

	trader_macd_div_threaded()
	trader_rsi_div_threaded()
	trader_stochastic_div_threaded()

	scheduler_trader_macd_div.add_job(func=trader_macd_div_threaded, trigger='cron', day_of_week=days, hour='00-23', minute=minute_trader, timezone='UTC')
	scheduler_trader_stochastic_div.add_job(func=trader_stochastic_div_threaded, trigger='cron', day_of_week=days, hour='00-23', minute=minute_trader, timezone='UTC')
	scheduler_trader_rsi_div.add_job(func=trader_rsi_div_threaded, trigger='cron', day_of_week=days, hour='00-23', minute=minute_trader, timezone='UTC')

	scheduler_news.add_job(func=news_threaded, trigger='cron', day_of_week='mon-fri', hour=hour_news, minute=minute_news, timezone='UTC')
	# Start the scheduler
	scheduler_trader_macd_div.start()
	scheduler_trader_stochastic_div.start()
	scheduler_trader_rsi_div.start()

	scheduler_news.start()

	try:
		asyncio.get_event_loop().run_forever()
	except (KeyboardInterrupt, SystemExit):
		pass

Run()