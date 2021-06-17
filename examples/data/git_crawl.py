from github import Github
from tqdm import tqdm
import time
import datetime
import os
import pickle
import numpy as np
import requests
import json

git = Github("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")


def get_timestamp(date):
    return int(time.mktime(time.strptime(str(date),'%Y-%m-%d %H:%M:%S')))

def get_commit_tuple(commit):
    id = None
    if commit.author:
        id = commit.author.id
    time = get_time(commit.commit.committer.date)
    return (id, time)

def crawl(func, max_count:int=None, **kwargs):
    n = func(**kwargs).totalCount
    print('total count:', n)
    
    if max_count is None:
        max_count = n
    res = [o for o in tqdm(func(**kwargs)[:max_count], total=max_count)]
    return res


def make_act_dict(users, times, act:dict=None):
    if act is None:
        act = {}
    for user, t in zip(users, times):
        if user is None:
            continue

        if user not in act:
            act[user] = []
        act[user] += [t]
    return act


def main(repo_name):
    repo = git.get_repo(repo_name)
    offset = get_timestamp(repo.created_at)
    
    releases = crawl(repo.get_releases)
    releases = list(map(lambda x: get_timestamp(x.created_at) - offset, releases))
    
    issues = crawl(repo.get_issues)
    issues = [get_timestamp(issue.created_at)-offset for issue in issues]
    
    stars = crawl(repo.get_stargazers_with_dates)
    stars = [get_timestamp(star.starred_at)-offset for star in stars]
    
    commits = crawl(repo.get_commits)
    commit_users = []
    commit_times = []
    for commit in commits:
        try:
            if commit.author is not None:
                commit_users += [commit.author.id]
                commit_times += [get_timestamp(commit.commit.committer.date)-offset]
        except:
            pass
    act = make_act_dict(commit_users, commit_times)
    
    forks = crawl(repo.get_forks)
    forks = [get_timestamp(fork.created_at)-offset for fork in forks]
    
    
    stats = {
        'forks': forks[::-1],
        'stars' : stars,
        'releases' : releases[::-1],
        'issues': issues[::-1],
        'activities': act
    }
    pickle.dump(stats, open(f"data/{'-'.join(repo_name.split('/'))}.pkl", 'wb'))
    
    
if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    repos = [
        'pytorch/pytorch', 'microsoft/terminal', 'microsoft/TypeScript', 'facebook/react' , 
        'facebook/react-native' , 'huggingface/transformers', 'apache/superset', 'apache/spark',
        'tensorflow/tensorflow', 'tensorflow/tensor2tensor', 'vuejs/vue', 'huggingface/datasets',
        'microsoft/vscode'
    ]
    for repo in repos:
        print(f'crawling {repo}...'
        main(repo)