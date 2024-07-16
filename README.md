# PPAudit

This is the repository for VPVet, a system that automatically vet privacy policy compliance issues in the VR ecosystem. Please refer to our paper ''VPVet: Vetting Privacy Policies of Virtual Reality Apps'' for more information.

## VRPP Dataset Collection
We use [WebScraper](https://webscraper.io/) to crawl meta-info (e.g., app name, genres, description, etc.) of VR apps from 10 different VR platforms, then we further use [Selenium](https://www.selenium.dev/) to crawl the privacy policies of VR apps from their privacy policies' links or homepages' links. For VR apps' APK files, there are two sources: Sidequest and Meta Quest. For SideQuest apps, we first fetch the apps' package URLs by[GetSidequestURL](https://github.com/mikeage/get_sidequest_urls) and then download them; while for the Meta Quest apps, we automate the downloading process by simultaneously controlling a rooted Android mobile phone (with our Meta Quest app installed and logging into our account) and the paired Quest device with a script running on PC.

![VR Platforms](https://github.com/Y-Zhan/PPAudit/blob/main/crawl_data/fig_vr_platforms.png)

