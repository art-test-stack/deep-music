# deep-music
ML program which generate music from tags

# env 

```bash
virtualenv -p python3.10 ~/envs/dmenv 
source ~/envs/dmenv/bin/activate
```

# spotify 

Follow the [Tuto](https://developer.spotify.com/documentation/web-api/tutorials/getting-started)

```bash
curl -X POST "https://accounts.spotify.com/api/token" \     
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "grant_type=client_credentials&client_id=$SPOTIFY_CLIENT_ID&client_secret=$SPOTIFY_CLIENT_SECRET"
```