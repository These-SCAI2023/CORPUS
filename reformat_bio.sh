#!/usr/bin/env bash

sed -i '' -E "s/([IB]);(PER|ORG|LOC|ORG|MISC)/\1-\2/g" "$1"

sed -i '' "s/;/ /g" "$1"
