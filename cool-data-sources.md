# Cool Datasets

## Kaggle Datasets

-   [New York AirBNB](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)
    -   id: listing ID
    -   name: name of the listing
    -   host\_id: host ID
    -   host\_name: name of the host
    -   neighbourhood\_group: location
    -   neighbourhood: area
    -   latitude: latitude coordinates
    -   longitude: longitude coordinates
    -   room\_type: listing space type
    -   price: price in dollars
    -   minimum\_nights: amount of nights minimum
    -   number\_of\_reviews: number of reviews
    -   last\_review: latest review
    -   reviews\_per\_month: number of reviews per month
    -   calculated\_host\_listings\_count: amount of listing per host
    -   availability\_365: number of days when listing is available for
        booking
-   [Ramen Reviews](https://www.kaggle.com/residentmario/ramen-ratings)
    -   Review \#
    -   Brand
    -   Variety
    -   Style
    -   Country
    -   Stars
    -   Top Ten
    -   group by country, style, agg stars
    -   how many top ten are null?
-   [Wine Reviews](https://www.kaggle.com/zynicide/wine-reviews)
-   [Google Play Store Apps](https://www.kaggle.com/lava18/google-play-store-apps)
-   [Kickstarter Projects](https://www.kaggle.com/kemical/kickstarter-projects)
-   [Used Cars](https://www.kaggle.com/austinreese/craigslist-carstrucks-data) (Careful), this one's a little over 1GB
-   [Boston Crimes](https://www.kaggle.com/AnalyzeBoston/crimes-in-boston)
-   [Several cool time series data sets](https://www.kaggle.com/shenba/time-series-datasets)
-   [Wikipedia movie reviews](https://www.kaggle.com/jrobischon/wikipedia-movie-plots)
-

## Other

-   [Zillow Data](https://www.zillow.com/research/data/)
-   [beareau of transportation statistics](https://www.transtats.bts.gov/OT\_Delay/OT\_DelayCause1.asp)
-   [border crossings](https://data.transportation.gov/Research-and-Statistics/Border-Crossing-Entry-Data/keg4-3bc2/data)

## Rdatasets

    python -m pip install pydataset

``` {.python}
from pydataset import data
```

``` {.python}
data('Fair') \# affair data

data('Housing')
```

``` {.python}
# * date. Month of data collection
# * psavert, personal savings rate, http://research.stlouisfed.org/fred2/series/PSAVERT/
# * pce, personal consumption expenditures, in billions of dollars, http://research.stlouisfed.org/fred2/series/PCE
# * unemploy, number of unemployed in thousands, http://research.stlouisfed.org/fred2/series/UNEMPLOY
# * uempmed, median duration of unemployment, in week, http://research.stlouisfed.org/fred2/series/UEMPMED
# * pop, total population, in thousands, http://research.stlouisfed.org/fred2/series/POP
df = data('economics')
df.date = pd.to_datetime(df.date)
df.set_index('date', inplace=True)
```

``` {.python}
# 2000 current population survey
# - state
# - year
# - vote: did they vote
# - income: 4 (< 5,000) to 17 (> 75000)
# - education: 1 (< high school diploma) through 4 (more than college)
# - age
# - female
df = data('voteincome').drop(columns='year')
```

``` {.python}
# * state state ID code
# * year year
# * mrall traffic fatality rate (deaths per 10000)
# * beertax tax on case of beer
# * mlda minimum legal drinking age
# * jaild mandatory jail sentence ?
# * comserd mandatory community service ?
# * vmiles average miles per driver
# * unrate unemployment rate
# * perinc per capita personal income
df = data('Fatality')
df.drop(df.query('vmiles > 18').index, inplace=True)
df.jaild = np.where(df.jaild == 'no', 0, 1)
df.comserd = np.where(df.comserd == 'no', 0, 1)
```

``` {.python}
# - region: U. S. Census regions. A factor with levels: `ENC`, East North
#   Central `ESC`, East South Central; `MA`, Mid-Atlantic; `MTN`, Mountain; `NE`,
#   New England; `PAC`, Pacific; `SA`, South Atlantic; `WNC`, West North Central;
#   `WSC`, West South Central.
# - pop: Population: in 1,000s.
# - SATV: Average score of graduating high-school students in the state on the
#   _verbal_ component of the Scholastic Aptitude Test (a standard university
#   admission exam).
# - SATM: Average score of graduating high-school students in the state on the
#   _math_ component of the Scholastic Aptitude Test.
# - percent: Percentage of graduating high-school students in the state who took
#   the SAT exam.
# - dollars: State spending on public education, in \$1000s per student.
# - pay: Average teacher's salary in the state, in $1000s.
df = data('States')
df['SAT'] = df.SATV + df.SATM
```

