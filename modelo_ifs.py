class ProgramCategorizer:
  def __init__(self):
    # Define diferentes expresiones regulares que indican si el nombre del porgrama puede pertenecer a alguna clase
    self.pattern_btb = re.compile(r'back[\s-]*to[\s-]*business|btb', flags=re.IGNORECASE)
    self.pattern_bts = re.compile(r'back[\s-]*to[\s-]*school|bts', flags=re.IGNORECASE)
    self.pattern_hol = re.compile(r'black[\s-]*friday|holiday|hol|newegg|egg[\s-]*diamond', flags=re.IGNORECASE)
    self.pattern_email = re.compile(r'email', flags=re.IGNORECASE)
    self.pattern_search = re.compile(r'search|sponsor[\s-]*product|sponsored[\s-]*product|google[\s-]*fund', flags=re.IGNORECASE)
    self.pattern_digital_0 = re.compile(r'digital[\s-]*advertising', flags=re.IGNORECASE)
    self.pattern_display = re.compile(r'\b(?!NEXCOM\b)(\w*online\w*|\w*banner\w*|\w*placement\w*|wall|\w*display\w*|\w*displays\w*|\w*carousel\w*|freight[\s-]*allowance|com|home[\s-]*page|homepage|retargeting|tech[\s-]*days|logic[\s-]*program)', flags=re.IGNORECASE)
    self.pattern_trad = re.compile(r'today[\s-]*ad|mail|usa[\s-]*today[\s-]*ads|magazine|circular|catalog|flyer|vfp|print[\s-]*ad|in[\s-]*print|rop', flags=re.IGNORECASE)
    self.pattern_digital = re.compile(r'digital', flags=re.IGNORECASE)
        

  def categorize_program(self, program_name):
    if isinstance(program_name, str):
      # categorizar un solo nombre de programa
      BTB_flag = bool(self.pattern_btb.search(program_name))
      BTS_flag = bool(self.pattern_bts.search(program_name))
      Hol_flag = bool(self.pattern_hol.search(program_name))
      Email_flag = bool(self.pattern_email.search(program_name))
      Search_flag = bool(self.pattern_search.search(program_name))
      Digital0_flag = bool(self.pattern_digital_0.search(program_name))
      Display_flag = bool(self.pattern_display.search(program_name))
      Trad_flag = bool(self.pattern_trad.search(program_name))
      Digital_flag = bool(self.pattern_digital.search(program_name))

      if BTB_flag:
        return 'BTB'
      elif BTS_flag:
        return 'BTS'
      elif Hol_flag:
        return 'Holiday'
      elif Email_flag:
        return 'Email'
      elif Search_flag:
        return 'Search'
      elif Digital0_flag:
        return 'Digital'
      elif Display_flag:
        return 'Display'
      elif Trad_flag:
        return 'Trad_media'
      elif Digital_flag:
        return 'Digital'
      else:
        return 'Program'
    elif isinstance(program_name, (np.ndarray, pd.Series, list)):
      # categorizar una matriz de nombres de programas
      df = pd.DataFrame({'program_name': program_name})
      df['BTB_flag'] = df['program_name'].apply(lambda x: bool(self.pattern_btb.search(x)))
      df['BTS_flag'] = df['program_name'].apply(lambda x: bool(self.pattern_bts.search(x)))
      df['Hol_flag'] = df['program_name'].apply(lambda x: bool(self.pattern_hol.search(x)))
      df['Email_flag'] = df['program_name'].apply(lambda x: bool(self.pattern_email.search(x)))
      df['Search_flag'] = df['program_name'].apply(lambda x: bool(self.pattern_search.search(x)))
      df['Digital0_flag'] = df['program_name'].apply(lambda x: bool(self.pattern_digital_0.search(x)))
      df['Display_flag'] = df['program_name'].apply(lambda x: bool(self.pattern_display.search(x)))
      df['Trad_flag'] = df['program_name'].apply(lambda x: bool(self.pattern_trad.search(x)))
      df['Digital_flag'] = df['program_name'].apply(lambda x: bool(self.pattern_digital.search(x)))

      # definir una función que devuelve la clase para cada fila
      def get_class(row):
        if row['BTB_flag']:
          return 'BTB'
        elif row['BTS_flag']:
          return 'BTS'
        elif row['Hol_flag']:
          return 'Holiday'
        elif row['Email_flag']:
          return 'Email'
        elif row['Search_flag']:
          return 'Search'
        elif row['Digital0_flag']:
          return 'Digital'
        elif row['Display_flag']:
          return 'Display'
        elif row['Trad_flag']:
          return 'Trad_media'
        elif row['Digital_flag']:
          return 'Digital'
        else:
          return 'Program'

      # aplicar la función para cada fila
      df['program_class'] = df.apply(get_class, axis=1)

      # devolver una serie con la clase de cada programa
      return df['program_class']
    else:
        raise ValueError('program_name debe ser una cadena o una matriz de cadenas.')
