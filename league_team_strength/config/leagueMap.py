import random
from datetime import datetime
import pyodbc

# Existing country data
existing_countries = {
    'EST': {'id': 2075994, 'name': 'Estonia'},
    'SVN': {'id': 2388631, 'name': 'Slovenia'},
    'BUL': {'id': 27349470, 'name': 'Bulgaria'},
    'UKR': {'id': 28614955, 'name': 'Ukraine'},
    'FRA': {'id': 50964166, 'name': 'France'},
    'KAZ': {'id': 82116667, 'name': 'Kazakhstan'},
    'LUX': {'id': 107154569, 'name': 'Luxembourg'},
    'TUR': {'id': 147650173, 'name': 'Türkiye'},
    'HUN': {'id': 202283776, 'name': 'Hungary'},
    'MDA': {'id': 202569033, 'name': 'Moldova'},
    'ROU': {'id': 209541860, 'name': 'Romania'},
    'ITA': {'id': 268951991, 'name': 'Italy'},
    'SVK': {'id': 292421301, 'name': 'Slovakia'},
    'ESP': {'id': 424224306, 'name': 'Spain'},
    'MLT': {'id': 454097546, 'name': 'Malta'},
    'POL': {'id': 491943898, 'name': 'Poland'},
    'BIH': {'id': 527518065, 'name': 'Bosnia and Herzegovina'},
    'IRL': {'id': 537678637, 'name': 'Republic of Ireland'},
    'AUT': {'id': 559669136, 'name': 'Austria'},
    'ARM': {'id': 685151039, 'name': 'Armenia'},
    'LVA': {'id': 694794026, 'name': 'Latvia'},
    'NED': {'id': 793229501, 'name': 'Netherlands'},
    'SRB': {'id': 804709685, 'name': 'Serbia'},
    'CZE': {'id': 837281983, 'name': 'Czechia'},
    'SUI': {'id': 986743702, 'name': 'Switzerland'},
    'ISL': {'id': 1079592624, 'name': 'Iceland'},
    'KOS': {'id': 1088812204, 'name': 'Kosovo'},
    'DEN': {'id': 1111626900, 'name': 'Denmark'},
    'NIR': {'id': 1219065687, 'name': 'Northern Ireland'},
    'ALB': {'id': 1290488319, 'name': 'Albania'},
    'MNE': {'id': 1318255058, 'name': 'Montenegro'},
    'SWE': {'id': 1371443743, 'name': 'Sweden'},
    'CRO': {'id': 1443712111, 'name': 'Croatia'},
    'ISR': {'id': 1515490351, 'name': 'Israel'},
    'NOR': {'id': 1585681674, 'name': 'Norway'},
    'RUS': {'id': 1609242252, 'name': 'Russia'},
    'GER': {'id': 1635415337, 'name': 'Germany'},
    'LTU': {'id': 1640991243, 'name': 'Lithuania'},
    'LIE': {'id': 1662559093, 'name': 'Liechtenstein'},
    'SCO': {'id': 1663582960, 'name': 'Scotland'},
    'ENG': {'id': 1693845392, 'name': 'England'},
    'BLR': {'id': 1698887797, 'name': 'Belarus'},
    'GIB': {'id': 1730504356, 'name': 'Gibraltar'},
    'MKD': {'id': 1769682597, 'name': 'North Macedonia'},
    'GRE': {'id': 1802602049, 'name': 'Greece'},
    'BEL': {'id': 1813411025, 'name': 'Belgium'},
    'WAL': {'id': 1859998060, 'name': 'Wales'},
    'FIN': {'id': 1870142565, 'name': 'Finland'},
    'AZE': {'id': 1903622339, 'name': 'Azerbaijan'},
    'CYP': {'id': 1928728568, 'name': 'Cyprus'},
    'POR': {'id': 1933320034, 'name': 'Portugal'},
    'GEO': {'id': 1940188566, 'name': 'Georgia'},
    'FRO': {'id': 1991806584, 'name': 'Faroe Islands'}
}

# Country mapping for leagues
country_mapping = {
    'Germany': ['bundesliga', '2.bundesliga', '3.liga', 'regionalliga', 'dfb_pokal'],
    'Austria': ['bundesliga_austria', '2.liga_austria', 'ofb_cup'],
    'Switzerland': ['super_league', 'challenge_league', 'swiss_cup'],
    'Belgium': ['pro_league', 'challenger_pro_league', 'belgian_cup'],
    'Netherlands': ['eredivisie', 'eerste_divisie', 'tweede_divisie', 'knvb_beker'],
    'Denmark': ['superliga', '1.division', 'dbupokalen'],
    'France': ['ligue_1', 'ligue_2', 'national', 'coupe_de_france'],
    'England': ['premier_league', 'championship', 'league_one', 'league_two', 'national_league', 'national_league_north_south', 'fa_cup', 'efl_cup'],
    'Italy': ['serie_a', 'serie_b', 'serie_c', 'coppa_italia'],
    'Spain': ['la_liga', 'segunda_division', 'primera_rfef', 'copa_del_rey'],
    'Portugal': ['primeira_liga', 'segunda_liga', 'liga_3', 'taca_de_portugal'],
    'Scotland': ['scottish_premiership', 'scottish_championship', 'scottish_league_one', 'scottish_league_two', 'scottish_cup'],
    'Sweden': ['allsvenskan', 'superettan', 'svenska_cupen'],
    'Norway': ['eliteserien', 'obos_ligaen', 'norsk_tipping_ligaen', 'PostNord-ligaen', 'nm_cup'],
    'Poland': ['ekstraklasa', '1_liga', '2_liga', 'puchar_polski'],
    'Slovakia': ['fortuna_liga', '2_liga_slovakia', 'slovensky_pohar'],
    'Czechia': ['czech_first_league', 'fnl'],
    'Greece': ['super_league_greece', 'super_league_2_greece', 'greek_cup'],
    'Türkiye': ['super_lig', '1_lig_turkey', 'turkish_cup'],
    'Russia': ['premier_league_russia', 'fnl_russia', 'russian_cup'],
    'Ukraine': ['premier_league_ukraine', 'ukrainian_cup'],
    'Croatia': ['hnl', 'croatian_cup'],
    'Serbia': ['super_liga_serbia', 'serbian_cup'],
    'Romania': ['liga_1_romania', 'liga_2_romania', 'cupa_romaniei'],
    'Hungary': ['nb1', 'magyar_kupa'],
    'Bulgaria': ['first_league_bulgaria', 'second_league_bulgaria', 'bulgarian_cup'],
    'Slovenia': ['prva_liga', 'slovenian_cup'],
    'Bosnia and Herzegovina': ['premijer_liga'],
    'North Macedonia': ['first_league_macedonia'],
    'Albania': ['kategoria_superiore'],
    'Montenegro': ['prva_crnogorska_liga'],
    'Northern Ireland': ['nifl_premiership', 'irish_cup'],
    'Wales': ['cymru_premier', 'welsh_cup'],
    'Belarus': ['vysshaya_liga', 'belarusian_cup'],
    'Lithuania': ['a_lyga'],
    'Latvia': ['virsliga'],
    'Estonia': ['meistriliiga'],
    'Moldova': ['super_liga_moldova', 'moldovan_cup'],
    'Cyprus': ['first_division_cyprus', 'second_division_cyprus', 'cypriot_cup'],
    'Luxembourg': ['bgl_ligue', 'luxembourg_cup'],
    'Faroe Islands': ['betri_deildin', 'faroese_cup'],
    'Armenia': ['armenian_premier_league'],
    'Azerbaijan': ['premyer_liqa'],
    'Georgia': ['erovnuli_liga'],
    'Kazakhstan': ['premier_league_kazakhstan'],
    'Republic of Ireland': ['league_of_ireland_premier', 'league_of_ireland_first', 'fai_cup'],
    'Finland': ['veikkausliiga', 'ykkonen', 'finnish_cup', 'ykkosliiga'],
    'Iceland': ['urvalsdeild', '1_deild', '2_deild', 'icelandic_cup'],
    'Europe': ['champions_league', 'champions_league_qualification', 'europa_league', 'europa_league_qualification', 'europa_conference_league', 'europa_conference_league_qualification'],
    'United States': ['mls', 'usl_championship', 'usl_league_one', 'us_open_cup', 'mls_next_pro'],
    'Canada': ['canadian_premier_league', 'canadian_championship'],
    'Mexico': ['liga_mx', 'liga_expansion'],
    'Costa Rica': ['primera_division_costa_rica'],
    'Guatemala': ['liga_nacional_guatemala'],
    'Honduras': ['liga_nacional_honduras'],
    'El Salvador': ['primera_division_el_salvador'],
    'Panama': ['liga_panamena'],
    'Brazil': ['brasileirao_serie_a', 'brasileirao_serie_b', 'brasileirao_serie_c', 'brasileirao_serie_d', 'copa_do_brasil'],
    'Argentina': ['liga_profesional', 'primera_nacional', 'copa_argentina'],
    'Uruguay': ['primera_division_uruguay', 'segunda_division_uruguay'],
    'Chile': ['primera_division_chile', 'primera_b_chile', 'copa_chile'],
    'Colombia': ['primera_a_colombia', 'primera_b_colombia', 'copa_colombia'],
    'Paraguay': ['primera_division_paraguay'],
    'Ecuador': ['serie_a_ecuador'],
    'Peru': ['primera_division_peru'],
    'Bolivia': ['primera_division_bolivia'],
    'Venezuela': ['primera_division_venezuela'],
    'South America': ['copa_libertadores', 'copa_libertadores_qual', 'copa_sudamericana'],
    'CONCACAF': ['central_american_cup', 'concacaf_champions_cup', 'leagues_cup']
}

# Country codes for new countries
country_codes = {
    'Europe': 'EUR',
    'United States': 'USA',
    'Canada': 'CAN',
    'Mexico': 'MEX',
    'Costa Rica': 'CRC',
    'Guatemala': 'GTM',
    'Honduras': 'HND',
    'El Salvador': 'SLV',
    'Panama': 'PAN',
    'Brazil': 'BRA',
    'Argentina': 'ARG',
    'Uruguay': 'URY',
    'Chile': 'CHI',
    'Colombia': 'COL',
    'Paraguay': 'PAR',
    'Ecuador': 'ECU',
    'Peru': 'PER',
    'Bolivia': 'BOL',
    'Venezuela': 'VEN',
    'South America': 'SAM',
    'CONCACAF': 'CON'
}

# League data structure from your code
leagues = {
    'bundesliga': {'id': 54, 'slug': 'bundesliga', 'name': 'Bundesliga', 'start_year': 2012},
    '2.bundesliga': {'id': 146, 'slug': '2-bundesliga', 'name': '2. Bundesliga', 'start_year': 2012},
    '3.liga': {'id': 208, 'slug': '3-liga', 'name': '3. Liga', 'start_year': 2012},
    'regionalliga': {'id': 512, 'slug': 'regionalliga', 'name': 'Regionalliga', 'start_year': 2012},
    'dfb_pokal': {'id': 209, 'slug': 'dfb-pokal', 'name': 'DFB-Pokal', 'start_year': 2012},
    'bundesliga_austria': {'id': 38, 'slug': 'bundesliga', 'name': 'Bundesliga (Austria)', 'start_year': 2012},
    '2.liga_austria': {'id': 119, 'slug': '2-liga', 'name': '2. Liga (Austria)', 'start_year': 2012},
    'ofb_cup': {'id': 278, 'slug': 'cup', 'name': 'ÖFB-Cup', 'start_year': 2012},
    'super_league': {'id': 69, 'slug': 'super-league', 'name': 'Swiss Super League', 'start_year': 2012},
    'challenge_league': {'id': 163, 'slug': 'challenge-league', 'name': 'Challenge League', 'start_year': 2012},
    'swiss_cup': {'id': 164, 'slug': 'cup', 'name': 'Swiss Cup', 'start_year': 2012},
    'pro_league': {'id': 40, 'slug': 'first-division', 'name': 'Jupiler Pro League', 'start_year': 2012},
    'challenger_pro_league': {'id': 264, 'slug': 'first-division-b', 'name': 'Challenger Pro League', 'start_year': 2012},
    'belgian_cup': {'id': 149, 'slug': 'cup', 'name': 'Belgian Cup', 'start_year': 2012},
    'eredivisie': {'id': 57, 'slug': 'eredivisie', 'name': 'Eredivisie', 'start_year': 2012},
    'eerste_divisie': {'id': 111, 'slug': 'eerste-divisie', 'name': 'Eerste Divisie', 'start_year': 2012},
    'tweede_divisie': {'id': 9195, 'slug': 'tweede-divisie', 'name': 'Tweede Divisie', 'start_year': 2012},
    'knvb_beker': {'id': 235, 'slug': 'knvb-cup', 'name': 'KNVB Beker', 'start_year': 2012},
    'superliga': {'id': 46, 'slug': 'superligaen', 'name': 'Danish Superliga', 'start_year': 2012},
    '1.division': {'id': 85, 'slug': '1-division', 'name': '1. Division', 'start_year': 2012},
    'dbupokalen': {'id': 242, 'slug': 'dbu-pokalen', 'name': 'DBU Pokalen', 'start_year': 2012},
    'ligue_1': {'id': 53, 'slug': 'ligue-1', 'name': 'Ligue 1', 'start_year': 2012},
    'ligue_2': {'id': 110, 'slug': 'ligue-2', 'name': 'Ligue 2', 'start_year': 2012},
    'national': {'id': 8970, 'slug': 'national', 'name': 'National (France)', 'start_year': 2012},
    'coupe_de_france': {'id': 134, 'slug': 'coupe-de-france', 'name': 'Coupe de France', 'start_year': 2012},
    'premier_league': {'id': 47, 'slug': 'premier-league', 'name': 'Premier League', 'start_year': 2012},
    'championship': {'id': 48, 'slug': 'championship', 'name': 'Championship', 'start_year': 2012},
    'league_one': {'id': 108, 'slug': 'league-one', 'name': 'League One', 'start_year': 2012},
    'league_two': {'id': 109, 'slug': 'league-two', 'name': 'League Two', 'start_year': 2012},
    'national_league': {'id': 117, 'slug': 'national-league', 'name': 'National League (England)', 'start_year': 2012},
    'national_league_north_south': {'id': 8944, 'slug': 'national-north-south', 'name': 'National League North/South (England)', 'start_year': 2012},
    'fa_cup': {'id': 132, 'slug': 'fa-cup', 'name': 'FA Cup', 'start_year': 2012},
    'efl_cup': {'id': 133, 'slug': 'efl-cup', 'name': 'EFL Cup', 'start_year': 2012},
    'serie_a': {'id': 55, 'slug': 'serie', 'name': 'Serie A', 'start_year': 2012},
    'serie_b': {'id': 86, 'slug': 'serie-b', 'name': 'Serie B', 'start_year': 2012},
    'serie_c': {'id': 147, 'slug': 'serie-c', 'name': 'Serie C (Italy)', 'start_year': 2012},
    'coppa_italia': {'id': 141, 'slug': 'coppa-italia', 'name': 'Coppa Italia', 'start_year': 2012},
    'la_liga': {'id': 87, 'slug': 'laliga', 'name': 'La Liga', 'start_year': 2012},
    'segunda_division': {'id': 140, 'slug': 'laliga2', 'name': 'Segunda División', 'start_year': 2012},
    'primera_rfef': {'id': 8968, 'slug': 'primera-federacion', 'name': 'Primera RFEF', 'start_year': 2012},
    'copa_del_rey': {'id': 138, 'slug': 'copa-del-rey', 'name': 'Copa del Rey', 'start_year': 2012},
    'primeira_liga': {'id': 61, 'slug': 'liga-portugal', 'name': 'Primeira Liga', 'start_year': 2012},
    'segunda_liga': {'id': 185, 'slug': 'liga-portugal-2', 'name': 'Segunda Liga (Portugal)', 'start_year': 2012},
    'liga_3': {'id': 9112, 'slug': 'liga-3', 'name': 'Liga 3 (Portugal)', 'start_year': 2012},
    'taca_de_portugal': {'id': 186, 'slug': 'taca-de-portugal', 'name': 'Taça de Portugal', 'start_year': 2012},
    'scottish_premiership': {'id': 64, 'slug': 'premiership', 'name': 'Scottish Premiership', 'start_year': 2012},
    'scottish_championship': {'id': 123, 'slug': 'championship', 'name': 'Scottish Championship', 'start_year': 2012},
    'scottish_league_one': {'id': 124, 'slug': 'league-one', 'name': 'Scottish League One', 'start_year': 2012},
    'scottish_league_two': {'id': 125, 'slug': 'league-two', 'name': 'Scottish League Two', 'start_year': 2012},
    'scottish_cup': {'id': 137, 'slug': 'scottish-cup', 'name': 'Scottish Cup', 'start_year': 2012},
    'allsvenskan': {'id': 67, 'slug': 'allsvenskan', 'name': 'Allsvenskan', 'start_year': 2012},
    'superettan': {'id': 168, 'slug': 'superettan', 'name': 'Superettan', 'start_year': 2012},
    'svenska_cupen': {'id': 171, 'slug': 'svenska-cupen', 'name': 'Svenska Cupen', 'start_year': 2012},
    'eliteserien': {'id': 59, 'slug': 'eliteserien', 'name': 'Eliteserien', 'start_year': 2012},
    'obos_ligaen': {'id': 203, 'slug': '1-divisjon', 'name': 'OBOS-ligaen', 'start_year': 2012},
    'norsk_tipping_ligaen': {'id': 205, 'slug': 'norsk-tipping-ligaen', 'name': '3. Division (Norway)', 'start_year': 2012},
    'PostNord-ligaen': {'id': 204, 'slug': 'postnord-ligaen', 'name': 'PostNord-ligaen', 'start_year': 2012},
    'nm_cup': {'id': 206, 'slug': 'cup', 'name': 'Norwegian Cup', 'start_year': 2012},
    'ekstraklasa': {'id': 196, 'slug': 'ekstraklasa', 'name': 'Ekstraklasa', 'start_year': 2012},
    '1_liga': {'id': 197, 'slug': 'i-liga', 'name': 'I Liga', 'start_year': 2012},
    '2_liga': {'id': 8935, 'slug': 'ii-liga', 'name': 'II Liga', 'start_year': 2012},
    'puchar_polski': {'id': 198, 'slug': 'puchar-polski', 'name': 'Puchar Polski', 'start_year': 2012},
    'fortuna_liga': {'id': 176, 'slug': '1-liga', 'name': 'Fortuna Liga', 'start_year': 2012},
    '2_liga_slovakia': {'id': 8973, 'slug': '2-liga', 'name': '2. Liga (Slovakia)', 'start_year': 2012},
    'slovensky_pohar': {'id': 177, 'slug': 'fa-cup', 'name': 'Slovenský pohár', 'start_year': 2012},
    'czech_first_league': {'id': 122, 'slug': '1-liga', 'name': 'Czech First League', 'start_year': 2012},
    'fnl': {'id': 253, 'slug': 'fnl', 'name': 'FNL (Czech National Football League)', 'start_year': 2012},
    'super_league_greece': {'id': 135, 'slug': 'super-league-1', 'name': 'Super League Greece', 'start_year': 2012},
    'super_league_2_greece': {'id': 8815, 'slug': 'super-league-2', 'name': 'Super League 2 (Greece)', 'start_year': 2012},
    'greek_cup': {'id': 145, 'slug': 'greece-cup', 'name': 'Greek Cup', 'start_year': 2012},
    'super_lig': {'id': 71, 'slug': 'super-lig', 'name': 'Süper Lig', 'start_year': 2012},
    '1_lig_turkey': {'id': 165, 'slug': '1-lig', 'name': '1. Lig (Turkey)', 'start_year': 2012},
    'turkish_cup': {'id': 151, 'slug': 'turkish-cup', 'name': 'Turkish Cup', 'start_year': 2012},
    'premier_league_russia': {'id': 63, 'slug': 'premier-league', 'name': 'Russian Premier League', 'start_year': 2012},
    'fnl_russia': {'id': 338, 'slug': '1-division', 'name': 'FNL (Russia)', 'start_year': 2012},
    'russian_cup': {'id': 193, 'slug': 'russian-cup', 'name': 'Russian Cup', 'start_year': 2012},
    'premier_league_ukraine': {'id': 441, 'slug': 'premier-league', 'name': 'Ukrainian Premier League', 'start_year': 2012},
    'ukrainian_cup': {'id': 442, 'slug': 'cup', 'name': 'Ukrainian Cup', 'start_year': 2012},
    'hnl': {'id': 252, 'slug': 'hnl', 'name': 'HNL (Croatian First League)', 'start_year': 2012},
    'croatian_cup': {'id': 275, 'slug': 'croatian-cup', 'name': 'Croatian Cup', 'start_year': 2012},
    'super_liga_serbia': {'id': 182, 'slug': 'super-liga', 'name': 'Serbian SuperLiga', 'start_year': 2012},
    'serbian_cup': {'id': 183, 'slug': 'cup', 'name': 'Serbian Cup', 'start_year': 2012},
    'liga_1_romania': {'id': 189, 'slug': 'liga-i', 'name': 'Liga I Romania', 'start_year': 2012},
    'liga_2_romania': {'id': 9113, 'slug': 'liga-ii', 'name': 'Liga II Romania', 'start_year': 2012},
    'cupa_romaniei': {'id': 190, 'slug': 'cupa-romaniei', 'name': 'Cupa României', 'start_year': 2012},
    'nb1': {'id': 212, 'slug': 'nb-i', 'name': 'NB I (Hungary)', 'start_year': 2012},
    'magyar_kupa': {'id': 213, 'slug': 'fa-cup', 'name': 'Magyar Kupa', 'start_year': 2012},
    'first_league_bulgaria': {'id': 270, 'slug': 'first-professional-league', 'name': 'First League (Bulgaria)', 'start_year': 2012},
    'second_league_bulgaria': {'id': 9096, 'slug': 'second-professional-league', 'name': 'Second League (Bulgaria)', 'start_year': 2012},
    'bulgarian_cup': {'id': 271, 'slug': 'cup', 'name': 'Bulgarian Cup', 'start_year': 2012},
    'prva_liga': {'id': 173, 'slug': 'prva-liga', 'name': 'Prva Liga Slovenia', 'start_year': 2012},
    'slovenian_cup': {'id': 174, 'slug': 'cup', 'name': 'Slovenian Cup', 'start_year': 2012},
    'premijer_liga': {'id': 267, 'slug': 'premier-league', 'name': 'Premijer Liga Bosnia', 'start_year': 2012},
    'first_league_macedonia': {'id': 249, 'slug': 'prva-liga', 'name': 'First League (North Macedonia)', 'start_year': 2012},
    'kategoria_superiore': {'id': 260, 'slug': 'kategoria-superiore', 'name': 'Kategoria Superiore', 'start_year': 2012},
    'prva_crnogorska_liga': {'id': 232, 'slug': '1-cfl', 'name': 'Prva Crnogorska Liga', 'start_year': 2012},
    'nifl_premiership': {'id': 129, 'slug': 'premiership', 'name': 'NIFL Premiership', 'start_year': 2012},
    'irish_cup': {'id': 9389, 'slug': 'irish-cup', 'name': 'Northern Irish Cup', 'start_year': 2012},
    'cymru_premier': {'id': 116, 'slug': 'cymru-premier', 'name': 'Cymru Premier', 'start_year': 2012},
    'welsh_cup': {'id': 9166, 'slug': 'welsh-cup', 'name': 'Welsh Cup', 'start_year': 2012},
    'vysshaya_liga': {'id': 263, 'slug': 'premier-league', 'name': 'Vysheyshaya Liga', 'start_year': 2012},
    'belarusian_cup': {'id': 9521, 'slug': 'cup', 'name': 'Belarusian Cup', 'start_year': 2012},
    'a_lyga': {'id': 228, 'slug': 'lyga', 'name': 'A Lyga', 'start_year': 2012},
    'virsliga': {'id': 226, 'slug': 'virsliga', 'name': 'Virsliga', 'start_year': 2012},
    'meistriliiga': {'id': 248, 'slug': 'premium-liiga', 'name': 'Meistriliiga', 'start_year': 2012},
    'super_liga_moldova': {'id': 231, 'slug': 'national-division', 'name': 'Super Liga (Moldova)', 'start_year': 2012},
    'moldovan_cup': {'id': 9530, 'slug': 'cup', 'name': 'Moldovan Cup', 'start_year': 2012},
    'first_division_cyprus': {'id': 136, 'slug': '1-division', 'name': 'First Division (Cyprus)', 'start_year': 2012},
    'second_division_cyprus': {'id': 9100, 'slug': '2-division', 'name': 'Second Division (Cyprus)', 'start_year': 2012},
    'cypriot_cup': {'id': 330, 'slug': 'cup', 'name': 'Cypriot Cup', 'start_year': 2012},
    'bgl_ligue': {'id': 229, 'slug': 'national-division', 'name': 'BGL Ligue', 'start_year': 2012},
    'luxembourg_cup': {'id': 9527, 'slug': 'cup', 'name': 'Luxembourg Cup', 'start_year': 2012},
    'betri_deildin': {'id': 250, 'slug': 'premier-league', 'name': 'Premier League Faroe Islands', 'start_year': 2012},
    'faroese_cup': {'id': 9523, 'slug': 'cup', 'name': 'Faroese Cup', 'start_year': 2015},
    'armenian_premier_league': {'id': 118, 'slug': 'premier-league', 'name': 'Armenian Premier League', 'start_year': 2012},
    'premyer_liqa': {'id': 262, 'slug': 'premier-league', 'name': 'Premyer Liqa', 'start_year': 2012},
    'erovnuli_liga': {'id': 439, 'slug': 'erovnuli-liga', 'name': 'Erovnuli Liga', 'start_year': 2012},
    'premier_league_kazakhstan': {'id': 225, 'slug': 'premier-league', 'name': 'Kazakhstan Premier League', 'start_year': 2012},
    'league_of_ireland_premier': {'id': 126, 'slug': 'premier-division', 'name': 'League of Ireland Premier Division', 'start_year': 2012},
    'league_of_ireland_first': {'id': 218, 'slug': 'first-division', 'name': 'League of Ireland First Division', 'start_year': 2012},
    'fai_cup': {'id': 219, 'slug': 'fai-cup', 'name': 'FAI Cup', 'start_year': 2012},
    'veikkausliiga': {'id': 51, 'slug': 'veikkausliiga', 'name': 'Veikkausliiga', 'start_year': 2012},
    'ykkonen': {'id': 8969, 'slug': 'ykkonen', 'name': 'Ykkönen', 'start_year': 2012},
    'finnish_cup': {'id': 143, 'slug': 'suomen-cup', 'name': 'Finnish Cup', 'start_year': 2012},
    'ykkosliiga': {'id': 251, 'slug': 'ykkosliiga', 'name': 'Ykkosliiga', 'start_year': 2012},
    'urvalsdeild': {'id': 215, 'slug': 'besta-deildin', 'name': 'Besta Deilden', 'start_year': 2012},
    '1_deild': {'id': 216, 'slug': '1-deild', 'name': '1. Deild', 'start_year': 2012},
    '2_deild': {'id': 10226, 'slug': '2-deild', 'name': '2. Deild', 'start_year': 2012},
    'icelandic_cup': {'id': 217, 'slug': 'icelandic-cup', 'name': 'Icelandic Cup', 'start_year': 2012},
    'champions_league': {'id': 42, 'slug': 'champions-league', 'name': 'Champions League', 'start_year': 2012},
    'champions_league_qualification': {'id': 10611, 'slug': 'champions-league-qualification', 'name': 'Champions League', 'start_year': 2012},
    'europa_league': {'id': 73, 'slug': 'europa-league', 'name': 'Europa League', 'start_year': 2012},
    'europa_league_qualification': {'id': 10613, 'slug': 'europa-league-qualification', 'name': 'Europa League', 'start_year': 2012},
    'europa_conference_league': {'id': 10216, 'slug': 'conference-league', 'name': 'Europa Conference League', 'start_year': 2021},
    'europa_conference_league_qualification': {'id': 10615, 'slug': 'conference-league-qualification', 'name': 'Europa Conference League', 'start_year': 2021},
    'mls': {'id': 130, 'slug': 'mls', 'name': 'MLS', 'start_year': 2012},
    'usl_championship': {'id': 8972, 'slug': 'usl-championship', 'name': 'USL Championship', 'start_year': 2012},
    'usl_league_one': {'id': 9296, 'slug': 'usl-league-one', 'name': 'USL League One', 'start_year': 2019},
    'us_open_cup': {'id': 9441, 'slug': 'open-cup', 'name': 'US Open Cup', 'start_year': 2012},
    'mls_next_pro': {'id': 10282, 'slug': 'mls-next-pro', 'name': 'MLS Next Pro', 'start_year': 2022},
    'canadian_premier_league': {'id': 9986, 'slug': 'premier-league', 'name': 'Canadian Premier League', 'start_year': 2019},
    'canadian_championship': {'id': 9837, 'slug': 'canadian-championship', 'name': 'Canadian Championship', 'start_year': 2018},
    'liga_mx': {'id': 230, 'slug': 'liga-mx', 'name': 'Liga MX', 'start_year': 2012},
    'liga_expansion': {'id': 10759, 'slug': 'liga-de-expansion-mx', 'name': 'Liga Expansión MX', 'start_year': 2020},
    'primera_division_costa_rica': {'id': 121, 'slug': 'primera-division', 'name': 'Costa Rica Primera División', 'start_year': 2012},
    'liga_nacional_guatemala': {'id': 336, 'slug': 'liga-nacional', 'name': 'Liga Nacional Guatemala', 'start_year': 2012},
    'liga_nacional_honduras': {'id': 337, 'slug': 'liga-nacional', 'name': 'Liga Nacional Honduras', 'start_year': 2012},
    'primera_division_el_salvador': {'id': 335, 'slug': 'primera-division', 'name': 'Primera División El Salvador', 'start_year': 2012},
    'liga_panamena': {'id': 10051, 'slug': 'lpf', 'name': 'Liga Panameña', 'start_year': 2023},
    'brasileirao_serie_a': {'id': 268, 'slug': 'serie', 'name': 'Brasileirão Série A', 'start_year': 2012},
    'brasileirao_serie_b': {'id': 8814, 'slug': 'serie-b', 'name': 'Brasileirão Série B', 'start_year': 2012},
    'brasileirao_serie_c': {'id': 8971, 'slug': 'serie-c', 'name': 'Brasileirão Série C', 'start_year': 2012},
    'brasileirao_serie_d': {'id': 9464, 'slug': 'serie-d', 'name': 'Brasileirão Série D', 'start_year': 2025},
    'copa_do_brasil': {'id': 9067, 'slug': 'cup', 'name': 'Copa do Brasil', 'start_year': 2012},
    'liga_profesional': {'id': 112, 'slug': 'liga-profesional', 'name': 'Liga Profesional Argentina', 'start_year': 2021},
    'primera_nacional': {'id': 8965, 'slug': 'primera-b-nacional', 'name': 'Primera B Nacional', 'start_year': 2021},
    'copa_argentina': {'id': 9305, 'slug': 'copa-argentina', 'name': 'Copa Argentina', 'start_year': 2015},
    'primera_division_uruguay': {'id': 161, 'slug': 'primera-division', 'name': 'Primera División Uruguay', 'start_year': 2016},
    'segunda_division_uruguay': {'id': 9122, 'slug': 'segunda-division', 'name': 'Segunda División Uruguay', 'start_year': 2016},
    'primera_division_chile': {'id': 273, 'slug': 'primera-division', 'name': 'Primera División Chile', 'start_year': 2017},
    'primera_b_chile': {'id': 10415, 'slug': 'primera-b', 'name': 'Primera B Chile', 'start_year': 2017},
    'copa_chile': {'id': 9091, 'slug': 'cup', 'name': 'Copa Chile', 'start_year': 2015},
    'primera_a_colombia': {'id': 274, 'slug': 'primera', 'name': 'Primera A Colombia', 'start_year': 2012},
    'primera_b_colombia': {'id': 9125, 'slug': 'primera-b', 'name': 'Primera B Colombia', 'start_year': 2012},
    'copa_colombia': {'id': 9490, 'slug': 'copa-colombia', 'name': 'Copa Colombia', 'start_year': 2015},
    'primera_division_paraguay': {'id': 199, 'slug': 'division-profesional', 'name': 'Primera División Paraguay', 'start_year': 2012},
    'serie_a_ecuador': {'id': 246, 'slug': 'serie', 'name': 'Serie A Ecuador', 'start_year': 2012},
    'primera_division_peru': {'id': 131, 'slug': 'liga-1', 'name': 'Liga 1 Peru', 'start_year': 2012},
    'primera_division_bolivia': {'id': 144, 'slug': 'primera-division', 'name': 'Primera División Bolivia', 'start_year': 2017},
    'primera_division_venezuela': {'id': 339, 'slug': 'primera-division', 'name': 'Primera División Venezuela', 'start_year': 2015},
    'copa_libertadores': {'id': 45, 'slug': 'copa-libertadores', 'name': 'Copa Libertadores', 'start_year': 2012},
    'copa_libertadores_qual': {'id': 10618, 'slug': 'copa-libertadores-qualification', 'name': 'Copa Libertadores', 'start_year': 2012},
    'central_american_cup': {'id': 9682, 'slug': 'concacaf-central-american-cup', 'name': 'Central American Cup', 'start_year': 2017},
    'copa_sudamericana': {'id': 299, 'slug': 'copa-sudamericana', 'name': 'Copa Sudamericana', 'start_year': 2012},
    'concacaf_champions_cup': {'id': 297, 'slug': 'concacaf-champions-cup', 'name': 'CONCACAF Champions Cup', 'start_year': 2018},
    'leagues_cup': {'id': 10043, 'slug': 'leagues-cup', 'name': 'Leagues Cup', 'start_year': 2019}
}

import hashlib

def generate_country_id(country_name):
    """Generate a unique ID using hash (same method as UEFA scraper)"""
    import hashlib
    hash_object = hashlib.md5(country_name.encode())
    country_id = int(hash_object.hexdigest()[:8], 16)
    country_id = country_id % 2000000000  # Keep it under 2 billion
    return country_id

# ============================================================================
# MANUAL DIVISION ASSIGNMENT
# ============================================================================
# Edit this dictionary to set the correct division level for each league
# Use: 1, 2, 3, 4, 5, etc. for divisions, 'Cup' for cup competitions, 
#      'Continental' for international competitions

division_assignments = {
    # Germany
    'bundesliga': 1,
    '2.bundesliga': 2,
    '3.liga': 3,
    'regionalliga': 4,
    'dfb_pokal': 'Cup',
    
    # Austria
    'bundesliga_austria': 1,
    '2.liga_austria': 2,
    'ofb_cup': 'Cup',
    
    # Switzerland
    'super_league': 1,
    'challenge_league': 2,
    'swiss_cup': 'Cup',
    
    # Belgium
    'pro_league': 1,
    'challenger_pro_league': 2,
    'belgian_cup': 'Cup',
    
    # Netherlands
    'eredivisie': 1,
    'eerste_divisie': 2,
    'tweede_divisie': 3,
    'knvb_beker': 'Cup',
    
    # Denmark
    'superliga': 1,
    '1.division': 2,
    'dbupokalen': 'Cup',
    
    # France
    'ligue_1': 1,
    'ligue_2': 2,
    'national': 3,
    'coupe_de_france': 'Cup',
    
    # England
    'premier_league': 1,
    'championship': 2,
    'league_one': 3,
    'league_two': 4,
    'national_league': 5,
    'national_league_north_south': 6,
    'fa_cup': 'Cup',
    'efl_cup': 'Cup',
    
    # Italy
    'serie_a': 1,
    'serie_b': 2,
    'serie_c': 3,
    'coppa_italia': 'Cup',
    
    # Spain
    'la_liga': 1,
    'segunda_division': 2,
    'primera_rfef': 3,
    'copa_del_rey': 'Cup',
    
    # Portugal
    'primeira_liga': 1,
    'segunda_liga': 2,
    'liga_3': 3,
    'taca_de_portugal': 'Cup',
    
    # Scotland
    'scottish_premiership': 1,
    'scottish_championship': 2,
    'scottish_league_one': 3,
    'scottish_league_two': 4,
    'scottish_cup': 'Cup',
    
    # Sweden
    'allsvenskan': 1,
    'superettan': 2,
    'svenska_cupen': 'Cup',
    
    # Norway
    'eliteserien': 1,
    'obos_ligaen': 2,
    'norsk_tipping_ligaen': 4,
    'PostNord-ligaen': 3,
    'nm_cup': 'Cup',
    
    # Poland
    'ekstraklasa': 1,
    '1_liga': 2,
    '2_liga': 3,
    'puchar_polski': 'Cup',
    
    # Slovakia
    'fortuna_liga': 1,
    '2_liga_slovakia': 2,
    'slovensky_pohar': 'Cup',
    
    # Czech Republic
    'czech_first_league': 1,
    'fnl': 2,
    
    # Greece
    'super_league_greece': 1,
    'super_league_2_greece': 2,
    'greek_cup': 'Cup',
    
    # Turkey
    'super_lig': 1,
    '1_lig_turkey': 2,
    'turkish_cup': 'Cup',
    
    # Russia
    'premier_league_russia': 1,
    'fnl_russia': 2,
    'russian_cup': 'Cup',
    
    # Ukraine
    'premier_league_ukraine': 1,
    'ukrainian_cup': 'Cup',
    
    # Croatia
    'hnl': 1,
    'croatian_cup': 'Cup',
    
    # Serbia
    'super_liga_serbia': 1,
    'serbian_cup': 'Cup',
    
    # Romania
    'liga_1_romania': 1,
    'liga_2_romania': 2,
    'cupa_romaniei': 'Cup',
    
    # Hungary
    'nb1': 1,
    'magyar_kupa': 'Cup',
    
    # Bulgaria
    'first_league_bulgaria': 1,
    'second_league_bulgaria': 2,
    'bulgarian_cup': 'Cup',
    
    # Slovenia
    'prva_liga': 1,
    'slovenian_cup': 'Cup',
    
    # Bosnia and Herzegovina
    'premijer_liga': 1,
    
    # North Macedonia
    'first_league_macedonia': 1,
    
    # Albania
    'kategoria_superiore': 1,
    
    # Montenegro
    'prva_crnogorska_liga': 1,
    
    # Northern Ireland
    'nifl_premiership': 1,
    'irish_cup': 'Cup',
    
    # Wales
    'cymru_premier': 1,
    'welsh_cup': 'Cup',
    
    # Belarus
    'vysshaya_liga': 1,
    'belarusian_cup': 'Cup',
    
    # Lithuania
    'a_lyga': 1,
    
    # Latvia
    'virsliga': 1,
    
    # Estonia
    'meistriliiga': 1,
    
    # Moldova
    'super_liga_moldova': 1,
    'moldovan_cup': 'Cup',
    
    # Cyprus
    'first_division_cyprus': 1,
    'second_division_cyprus': 2,
    'cypriot_cup': 'Cup',
    
    # Luxembourg
    'bgl_ligue': 1,
    'luxembourg_cup': 'Cup',
    
    # Faroe Islands
    'betri_deildin': 1,
    'faroese_cup': 'Cup',
    
    # Armenia
    'armenian_premier_league': 1,
    
    # Azerbaijan
    'premyer_liqa': 1,
    
    # Georgia
    'erovnuli_liga': 1,
    
    # Kazakhstan
    'premier_league_kazakhstan': 1,
    
    # Republic of Ireland
    'league_of_ireland_premier': 1,
    'league_of_ireland_first': 2,
    'fai_cup': 'Cup',
    
    # Finland
    'veikkausliiga': 1,
    'ykkonen': 2,
    'finnish_cup': 'Cup',
    'ykkosliiga': 3,
    
    # Iceland
    'urvalsdeild': 1,
    '1_deild': 2,
    '2_deild': 3,
    'icelandic_cup': 'Cup',
    
    # Continental - Europe
    'champions_league': 'Continental',
    'champions_league_qualification': 'Continental',
    'europa_league': 'Continental',
    'europa_league_qualification': 'Continental',
    'europa_conference_league': 'Continental',
    'europa_conference_league_qualification': 'Continental',
    
    # United States
    'mls': 1,
    'usl_championship': 2,
    'usl_league_one': 3,
    'us_open_cup': 'Cup',
    'mls_next_pro': 4,
    
    # Canada
    'canadian_premier_league': 1,
    'canadian_championship': 'Cup',
    
    # Mexico
    'liga_mx': 1,
    'liga_expansion': 2,
    
    # Costa Rica
    'primera_division_costa_rica': 1,
    
    # Guatemala
    'liga_nacional_guatemala': 1,
    
    # Honduras
    'liga_nacional_honduras': 1,
    
    # El Salvador
    'primera_division_el_salvador': 1,
    
    # Panama
    'liga_panamena': 1,
    
    # Brazil
    'brasileirao_serie_a': 1,
    'brasileirao_serie_b': 2,
    'brasileirao_serie_c': 3,
    'brasileirao_serie_d': 4,
    'copa_do_brasil': 'Cup',
    
    # Argentina
    'liga_profesional': 1,
    'primera_nacional': 2,
    'copa_argentina': 'Cup',
    
    # Uruguay
    'primera_division_uruguay': 1,
    'segunda_division_uruguay': 2,
    
    # Chile
    'primera_division_chile': 1,
    'primera_b_chile': 2,
    'copa_chile': 'Cup',
    
    # Colombia
    'primera_a_colombia': 1,
    'primera_b_colombia': 2,
    'copa_colombia': 'Cup',
    
    # Paraguay
    'primera_division_paraguay': 1,
    
    # Ecuador
    'serie_a_ecuador': 1,
    
    # Peru
    'primera_division_peru': 1,
    
    # Bolivia
    'primera_division_bolivia': 1,
    
    # Venezuela
    'primera_division_venezuela': 1,
    
    # Continental - South America
    'copa_libertadores': 'Continental',
    'copa_libertadores_qual': 'Continental',
    'copa_sudamericana': 'Continental',
    
    # Continental - CONCACAF
    'central_american_cup': 'Continental',
    'concacaf_champions_cup': 'Continental',
    'leagues_cup': 'Continental',
}

def sort_division(div):
    if div == 'Cup':
        return (1, 1000, div)
    elif div == 'Continental':
        return (1, 2000, div)
    elif div == 'UNASSIGNED':
        return (1, 3000, div)
    else:
        try:
            return (0, int(div), '')
        except (ValueError, TypeError):
            return (1, 9999, div)
        
def get_division_level(league_key, league_name):
    """Get the division level from the manual assignments dictionary"""
    return division_assignments.get(league_key, 'UNASSIGNED')

# TASK 1: Create country IDs for missing countries
print("=" * 80)
print("TASK 1: NEW COUNTRIES TO ADD")
print("=" * 80)

new_countries = []
for country in country_mapping.keys():
    # Try to find existing country by name
    existing = None
    for code, data in existing_countries.items():
        if data['name'] == country:
            existing = code
            break
    
    if not existing:
        country_code = country_codes.get(country, country[:3].upper())
        country_id = generate_country_id(country)  # Use hash instead of random
        new_countries.append({
            'id': country_id,
            'name': country,
            'code': country_code,
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        })

print(f"\nFound {len(new_countries)} new countries to add:\n")
print("CountryID\tCountryName\tCountryCode\tCreatedDate")
print("-" * 80)
for country in new_countries:
    print(f"{country['id']}\t{country['name']}\t{country['code']}\t{country['created_date']}")

# Combine all countries
all_countries = dict(existing_countries)
for new_country in new_countries:
    all_countries[new_country['code']] = {
        'id': new_country['id'],
        'name': new_country['name']
    }

# Create reverse lookup (name to code)
name_to_code = {data['name']: code for code, data in all_countries.items()}

# TASK 2: Create league-country lookup table
print("\n" + "=" * 80)
print("TASK 2: LEAGUE-COUNTRY LOOKUP TABLE")
print("=" * 80)

league_country_lookup = []
for country, league_keys in country_mapping.items():
    if country in name_to_code:
        country_code = name_to_code[country]
        country_id = all_countries[country_code]['id']
        
        for league_key in league_keys:
            if league_key in leagues:
                league_country_lookup.append({
                    'league_key': league_key,
                    'league_id': leagues[league_key]['id'],
                    'league_name': leagues[league_key]['name'],
                    'country_id': country_id,
                    'country_name': country,
                    'country_code': country_code
                })

print(f"\nCreated lookup for {len(league_country_lookup)} leagues\n")
print("LeagueKey\tLeagueID\tLeagueName\tCountryID\tCountryName\tCountryCode")
print("-" * 120)
for entry in league_country_lookup[:20]:  # Show first 20
    print(f"{entry['league_key']}\t{entry['league_id']}\t{entry['league_name']}\t{entry['country_id']}\t{entry['country_name']}\t{entry['country_code']}")
print(f"... and {len(league_country_lookup) - 20} more")

# TASK 3: Create division table
print("\n" + "=" * 80)
print("TASK 3: LEAGUE DIVISION TABLE")
print("=" * 80)

# Define sort function at the beginning of this section
def sort_division(div):
    if div == 'Cup':
        return (1, 1000, div)
    elif div == 'Continental':
        return (1, 2000, div)
    elif div == 'UNASSIGNED':
        return (1, 3000, div)
    else:
        try:
            return (0, int(div), '')
        except (ValueError, TypeError):
            return (1, 9999, div)

division_table = []
unassigned_leagues = []

for entry in league_country_lookup:
    division_level = get_division_level(entry['league_key'], entry['league_name'])
    
    if division_level == 'UNASSIGNED':
        unassigned_leagues.append({
            'league_key': entry['league_key'],
            'league_name': entry['league_name'],
            'country': entry['country_name']
        })
    
    division_table.append({
        'country_id': entry['country_id'],
        'country_name': entry['country_name'],
        'league_name': entry['league_name'],
        'league_id': entry['league_id'],
        'division_level': division_level
    })

# Check for unassigned leagues
if unassigned_leagues:
    print("\n⚠ WARNING: Found unassigned leagues!")
    print("=" * 80)
    print("Please add these to the division_assignments dictionary:\n")
    for league in unassigned_leagues:
        print(f"    '{league['league_key']}': 1,  # {league['league_name']} ({league['country']})")
    print("\n" + "=" * 80)
    print("Script will continue, but these leagues will be marked as 'UNASSIGNED'")
    print("=" * 80)

# Sort by country and division
division_table.sort(key=lambda x: (
    x['country_name'], 
    sort_division(x['division_level'])[1]  # Use the numeric part of our sort function
))

print(f"\nCreated division table with {len(division_table)} entries\n")
print("CountryID\tCountryName\tLeagueName\tLeagueID\tDivision/Cup")
print("-" * 120)

current_country = None
for entry in division_table:
    if entry['country_name'] != current_country:
        if current_country is not None:
            print("-" * 120)
        current_country = entry['country_name']
    
    print(f"{entry['country_id']}\t{entry['country_name']}\t{entry['league_name']}\t{entry['league_id']}\t{entry['division_level']}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"Total countries: {len(all_countries)}")
print(f"New countries added: {len(new_countries)}")
print(f"Total leagues: {len(league_country_lookup)}")

division_counts = {}
for entry in division_table:
    div = str(entry['division_level'])
    division_counts[div] = division_counts.get(div, 0) + 1

print(f"\nLeagues by division:")
for div in sorted(division_counts.keys(), key=sort_division):
    print(f"  Division {div}: {division_counts[div]}")

# ============================================================================
# DATABASE UPLOAD SECTION
# ============================================================================

print("\n" + "=" * 80)
print("DATABASE CONNECTION AND UPLOAD")
print("=" * 80)

# Database connection parameters
# Update these with your actual connection details
server = 'DESKTOP-J9IV3OH'  # e.g., 'localhost' or 'SERVER\\SQLEXPRESS'
database = 'fussballDB'

# Connection string
# For Windows Authentication (trusted connection):
conn_str_trusted = (
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={server};'
    f'DATABASE={database};'
    f'Trusted_Connection=yes;'
)

def upload_to_sql():
    """Upload all data to SQL Server"""
    try:
        # Choose the appropriate connection string
        # Use conn_str_trusted for Windows Auth or conn_str_sql for SQL Auth
        conn = pyodbc.connect(conn_str_trusted)
        cursor = conn.cursor()
        
        print("\n✓ Successfully connected to database!")
        
        # ========================================================================
        # 0. REFRESH COUNTRY IDS FROM DATABASE
        # ========================================================================
        print("\n" + "-" * 80)
        print("STEP 0: Loading existing countries from database...")
        print("-" * 80)
        
        cursor.execute("SELECT CountryID, CountryName, CountryCode FROM Countries")
        db_countries = cursor.fetchall()
        
        # Create a lookup of existing countries by name
        existing_country_lookup = {}
        for row in db_countries:
            country_id, country_name, country_code = row
            existing_country_lookup[country_name] = {
                'id': country_id,
                'code': country_code
            }
        
        print(f"✓ Loaded {len(existing_country_lookup)} existing countries")
        
        # Update our all_countries dict with actual database IDs
        updated_all_countries = {}
        for code, data in all_countries.items():
            country_name = data['name']
            if country_name in existing_country_lookup:
                # Use existing ID from database
                updated_all_countries[code] = {
                    'id': existing_country_lookup[country_name]['id'],
                    'name': country_name
                }
                print(f"  ✓ Found existing: {country_name} -> ID {existing_country_lookup[country_name]['id']}")
            else:
                # Keep generated ID for new countries
                updated_all_countries[code] = data
        
        # Update name_to_code for the corrected IDs
        updated_name_to_code = {data['name']: code for code, data in updated_all_countries.items()}
        
        # Rebuild league_country_lookup with corrected country IDs
        updated_league_country_lookup = []
        for country, league_keys in country_mapping.items():
            if country in updated_name_to_code:
                country_code = updated_name_to_code[country]
                country_id = updated_all_countries[country_code]['id']
                
                for league_key in league_keys:
                    if league_key in leagues:
                        updated_league_country_lookup.append({
                            'league_key': league_key,
                            'league_id': leagues[league_key]['id'],
                            'league_name': leagues[league_key]['name'],
                            'country_id': country_id,
                            'country_name': country,
                            'country_code': country_code
                        })
        
        # Rebuild division_table with corrected country IDs
        updated_division_table = []
        for entry in updated_league_country_lookup:
            division_level = get_division_level(entry['league_key'], entry['league_name'])
            updated_division_table.append({
                'country_id': entry['country_id'],
                'country_name': entry['country_name'],
                'league_name': entry['league_name'],
                'league_id': entry['league_id'],
                'division_level': division_level
            })
        
        # Filter new_countries to only include those not in database
        countries_to_insert = [c for c in new_countries if c['name'] not in existing_country_lookup]
        
        print(f"✓ {len(countries_to_insert)} new countries need to be inserted")
        
        # ========================================================================
        # 1. INSERT NEW COUNTRIES
        # ========================================================================
        print("\n" + "-" * 80)
        print("STEP 1: Inserting new countries...")
        print("-" * 80)
        
        countries_inserted = 0
        for country in countries_to_insert:
            try:
                insert_query = """
                INSERT INTO Countries (CountryID, CountryName, CountryCode, CreatedDate)
                VALUES (?, ?, ?, ?)
                """
                cursor.execute(insert_query, 
                             country['id'], 
                             country['name'], 
                             country['code'], 
                             country['created_date'])
                countries_inserted += 1
                print(f"  ✓ Inserted: {country['name']} ({country['code']})")
            except pyodbc.IntegrityError as e:
                print(f"  ⚠ Skipped {country['name']}: Already exists or constraint violation")
            except Exception as e:
                print(f"  ✗ Error inserting {country['name']}: {str(e)}")
        
        conn.commit()
        print(f"\n✓ Inserted {countries_inserted} new countries")
        
        # ========================================================================
        # 2. CREATE AND POPULATE LEAGUE_COUNTRY_LOOKUP TABLE
        # ========================================================================
        print("\n" + "-" * 80)
        print("STEP 2: Creating LeagueCountryLookup table...")
        print("-" * 80)
        
        # Create table if it doesn't exist
        create_lookup_table = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='LeagueCountryLookup' AND xtype='U')
        CREATE TABLE LeagueCountryLookup (
            LeagueKey NVARCHAR(100) PRIMARY KEY,
            LeagueID INT NOT NULL,
            LeagueName NVARCHAR(255) NOT NULL,
            CountryID INT NOT NULL,
            CountryName NVARCHAR(100) NOT NULL,
            CountryCode NVARCHAR(10) NOT NULL,
            FOREIGN KEY (CountryID) REFERENCES Countries(CountryID)
        )
        """
        cursor.execute(create_lookup_table)
        conn.commit()
        print("✓ LeagueCountryLookup table ready")
        
        # Clear existing data
        cursor.execute("DELETE FROM LeagueCountryLookup")
        conn.commit()
        
        # Insert lookup data with corrected IDs
        lookup_inserted = 0
        for entry in updated_league_country_lookup:
            try:
                insert_query = """
                INSERT INTO LeagueCountryLookup 
                (LeagueKey, LeagueID, LeagueName, CountryID, CountryName, CountryCode)
                VALUES (?, ?, ?, ?, ?, ?)
                """
                cursor.execute(insert_query,
                             entry['league_key'],
                             entry['league_id'],
                             entry['league_name'],
                             entry['country_id'],
                             entry['country_name'],
                             entry['country_code'])
                lookup_inserted += 1
            except Exception as e:
                print(f"  ✗ Error inserting {entry['league_key']}: {str(e)}")
        
        conn.commit()
        print(f"✓ Inserted {lookup_inserted} league-country mappings")
        
        # ========================================================================
        # 3. CREATE AND POPULATE LEAGUE_DIVISIONS TABLE
        # ========================================================================
        print("\n" + "-" * 80)
        print("STEP 3: Creating LeagueDivisions table...")
        print("-" * 80)
        
        # Create table if it doesn't exist
        create_divisions_table = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='LeagueDivisions' AND xtype='U')
        CREATE TABLE LeagueDivisions (
            ID INT IDENTITY(1,1) PRIMARY KEY,
            CountryID INT NOT NULL,
            CountryName NVARCHAR(100) NOT NULL,
            LeagueName NVARCHAR(255) NOT NULL,
            LeagueID INT NOT NULL,
            DivisionLevel NVARCHAR(50) NOT NULL,
            FOREIGN KEY (CountryID) REFERENCES Countries(CountryID)
        )
        """
        cursor.execute(create_divisions_table)
        conn.commit()
        print("✓ LeagueDivisions table ready")
        
        # Clear existing data
        cursor.execute("DELETE FROM LeagueDivisions")
        conn.commit()
        
        # Insert division data with corrected IDs
        divisions_inserted = 0
        for entry in updated_division_table:
            try:
                insert_query = """
                INSERT INTO LeagueDivisions 
                (CountryID, CountryName, LeagueName, LeagueID, DivisionLevel)
                VALUES (?, ?, ?, ?, ?)
                """
                cursor.execute(insert_query,
                             entry['country_id'],
                             entry['country_name'],
                             entry['league_name'],
                             entry['league_id'],
                             str(entry['division_level']))
                divisions_inserted += 1
            except Exception as e:
                print(f"  ✗ Error inserting division for {entry['league_name']}: {str(e)}")
        
        conn.commit()
        print(f"✓ Inserted {divisions_inserted} league division records")
        
        # ========================================================================
        # VERIFICATION
        # ========================================================================
        print("\n" + "=" * 80)
        print("VERIFICATION - Database Contents")
        print("=" * 80)
        
        # Count countries
        cursor.execute("SELECT COUNT(*) FROM Countries")
        total_countries = cursor.fetchone()[0]
        print(f"\nTotal countries in database: {total_countries}")
        
        # Count lookups
        cursor.execute("SELECT COUNT(*) FROM LeagueCountryLookup")
        total_lookups = cursor.fetchone()[0]
        print(f"Total league-country mappings: {total_lookups}")
        
        # Count divisions
        cursor.execute("SELECT COUNT(*) FROM LeagueDivisions")
        total_divisions = cursor.fetchone()[0]
        print(f"Total league division records: {total_divisions}")
        
        # Show sample data from each table
        print("\n" + "-" * 80)
        print("Sample: New Countries (first 5)")
        print("-" * 80)
        cursor.execute("""
            SELECT TOP 5 CountryID, CountryName, CountryCode 
            FROM Countries 
            ORDER BY CreatedDate DESC
        """)
        for row in cursor.fetchall():
            print(f"  {row[0]}\t{row[1]}\t{row[2]}")
        
        print("\n" + "-" * 80)
        print("Sample: League-Country Lookup (first 5)")
        print("-" * 80)
        cursor.execute("""
            SELECT TOP 5 LeagueKey, LeagueName, CountryName 
            FROM LeagueCountryLookup
        """)
        for row in cursor.fetchall():
            print(f"  {row[0]}\t{row[1]}\t{row[2]}")
        
        print("\n" + "-" * 80)
        print("Sample: League Divisions by Country (first 10)")
        print("-" * 80)
        cursor.execute("""
            SELECT TOP 10 CountryName, LeagueName, DivisionLevel 
            FROM LeagueDivisions 
            ORDER BY CountryName, 
                CASE 
                    WHEN DivisionLevel = 'Cup' THEN 1000
                    WHEN DivisionLevel = 'Continental' THEN 2000
                    ELSE CAST(DivisionLevel AS INT)
                END
        """)
        for row in cursor.fetchall():
            print(f"  {row[0]}\t{row[1]}\t{row[2]}")
        
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 80)
        print("✓ DATABASE UPLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except pyodbc.Error as e:
        print(f"\n✗ Database connection error: {str(e)}")
        print("\nPlease check:")
        print("  1. Server name is correct")
        print("  2. Database name is correct")
        print("  3. You have the correct permissions")
        print("  4. SQL Server is running")
        print("  5. ODBC Driver 17 for SQL Server is installed")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        return False
    
    return True

# UNCOMMENT THE LINE BELOW TO RUN THE UPLOAD
upload_to_sql()