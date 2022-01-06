# Live Updates

This folder contains scripts and utilities that need to executed once a day to keep Tutela up to date. We need to update the Tornado Cash heuristics, the deposit address reuse algorithm, and Diff2Vec. These updates need to execute fast (< 1 hr). There are a lot of moving steps to updating Tutela. We will write a large bash script to do this. It will be run using CRON daily.

## Notes

Do not import from `live` outside of the `live` folder. This is meant to be a standalone folder, even if we have to repeat code.