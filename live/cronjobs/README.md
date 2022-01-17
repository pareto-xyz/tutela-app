# CRON jobs to run the live updating scripts

Helpful link [here](https://www.cyberciti.biz/faq/how-do-i-add-jobs-to-cron-under-linux-or-unix-oses/).

We will be using crontab. Each line of the crontab file defines a job. 

The syntax is 
```
1 2 3 4 5 /path/to/command arg1 arg2
```
where
```
1: Minute (0-59)
2: Hours (0-23)
3: Day (0-31)
4: Month (0-12 [12 == December])
5: Day of the week(0-7 [7 or 0 == sunday])
/path/to/command – Script or command name to schedule
```
Put `*` for where you want it run all the time.

We put the lines to add in `schedule.txt`. Note that all the files in this folder are customized to AWS-EC2. If you are using something else, you will need to change these files.
