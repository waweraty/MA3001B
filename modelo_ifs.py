class ProgramCategorizer:
  def __init__(self):
    # Define diferentes expresiones regulares que indican si el nombre del porgrama puede pertenecer a alguna clase
    # Las reglas que son tuplas de longitud 4 se utilizan para agregar reglas de menor jerarquia a una categoría que ya tenía reglas.
    self.rules = [
    ('BTB', 'BTB', re.compile(r'back[\s-]*to[\s-]*business|btb', flags=re.IGNORECASE)),
    ('BTS', 'BTS', re.compile(r'back[\s-]*to[\s-]*school|bts', flags=re.IGNORECASE)),
    ('Holiday', 'Holiday', re.compile(r'holiday|\bhol\b', flags=re.IGNORECASE)),
    ('Holiday', 'Black_Friday', re.compile(r'black[\s-]*friday', flags=re.IGNORECASE)),
    ('Email', 'Email', re.compile(r'\w*email\w*', flags=re.IGNORECASE)),
    ('Search', 'Sponsored_product', re.compile(r'sponsor[\s-]*product|sponsored[\s-]*product|logic', flags=re.IGNORECASE)),
    ('Search', 'pla', re.compile(r'PLA')),
    ('Search', 'Search', re.compile(r'search|google[\s-]*fund|google[\s-]*campaign', flags=re.IGNORECASE)),
    ('Trad_media', 'Circular', re.compile(r'circular|mail|flyer|vfp', flags=re.IGNORECASE)),
    ('Trad_media', 'Radio', re.compile(r'radio', flags=re.IGNORECASE)),
    ('Trad_media', 'Print', re.compile(r'catalog|magazine|usa[\s-]*today[\s-]*ad|print[\s-]*ad|in[\s-]*print', flags=re.IGNORECASE)),
    ('Trad_media', 'Print', re.compile(r'\w*ROP\w*')),
    ('Display', 'Banner', re.compile(r'\w*banner\w*', flags=re.IGNORECASE)),
    ('Display', 'Social', re.compile(r'facebook|social|twitter|tiktok|instagram', flags=re.IGNORECASE)),
    ('Display', 'Display', re.compile(r'\b(?!NEXCOM\b)(\w*amazon\w*|\w*online\w*|\w*placement\w*|wall|\w*display\w*|\w*displays\w*|\w*carousel\w*|freight[\s-]*allowance|com|home[\s-]*page|\w*homepage\w*|tech[\s-]*days|\w*retargeting\w*|demand[\s-]*generation|ad[\s-]*support|\w*dotcom\w*|dot[\s-]*com|\w*adsupport\w*)', flags=re.IGNORECASE)),
    ('Digital', 'Digital', re.compile(r'digital', flags=re.IGNORECASE)),
    ('Program', 'Endcap', re.compile(r'end[\s-]*cap', flags=re.IGNORECASE)),
    ('Program', 'Vendor', re.compile(r'vendor', flags=re.IGNORECASE)),
    ('Program', 'Billboard', re.compile(r'billboard|blbd', flags=re.IGNORECASE)),
    ('Program', 'Showcase', re.compile(r'showcase', flags=re.IGNORECASE)),
    ('Program', 'Amplification', re.compile(r'amplification', flags=re.IGNORECASE)),
    ('Program', 'Brand_building', re.compile(r'\b(?!Brandsmart\b)(brand)', flags=re.IGNORECASE)),
    ('Program', 'TIN', re.compile(r'TIN')),
    ('Program', 'Rewards', re.compile(r'reward|loyalty', flags=re.IGNORECASE)),
    ('Program', 'Choice', re.compile(r'choice', flags=re.IGNORECASE)),
    ('Program', 'Landing_page', re.compile(r'landing[\s-]*page', flags=re.IGNORECASE)),
    ('Program', 'Launch', re.compile(r'launch', flags=re.IGNORECASE)),
    ('Program', 'Advertising', re.compile(r'\w*fund\w*|mktg|marketing|advertising|\bad\b|campaign|adoption|ads|advertisement|adapter', flags=re.IGNORECASE)),
    ('Display', 'Display', re.compile(r'\w*amazon\w*', flags=re.IGNORECASE), 2)]
  def categorize_program(self, program_name):
    if isinstance(program_name, str):
      # categorizar un solo nombre de programa
      

      for i in self.rules:
        if bool(i[2].search(program_name)):
          return i[0:2]
      return ('Program', 'Other')


    elif isinstance(program_name, (np.ndarray, pd.Series, list)):
      # categorizar una serie de nombres de programas
      df = pd.DataFrame({'program_name': program_name})
      df = df.fillna('nan')
      for rule in self.rules:
        if len(rule) == 3:
          nombre = rule[0] + '*' + rule[1] + '_flag'
          if nombre in df.columns:
            df[nombre] = df[nombre] | df['program_name'].apply(lambda x: bool(rule[2].search(x)))
          else:
            df[nombre] = df['program_name'].apply(lambda x: bool(rule[2].search(x)))
        else:
          nombre = rule[0] + '*' + rule[1] + '_flag_' + str(rule[3])
          df[nombre] = df['program_name'].apply(lambda x: bool(rule[2].search(x)))


      # definir una función que devuelve la clase para cada fila
      def get_class(row):

        for rule in self.rules:
          if len(rule) == 3:
            nombre = rule[0] + '*' + rule[1] + '_flag'
            if row[nombre]:
              return nombre[:-5]
          else:
            nombre = rule[0] + '*' + rule[1] + '_flag_' + str(rule[3])
            if row[nombre]:
              return nombre[:-7]
        return 'Program*Other'

      # aplicar la función para cada fila
      df['pred'] = df.apply(get_class, axis=1)
      df[['activity_subtype', 'activity_subtype_id']] = df['pred'].str.split('*', expand=True)

      # devolver una serie con la clase de cada programa
      return df[['activity_subtype', 'activity_subtype_id']]
    else:
        raise ValueError('program_name debe ser una cadena o una matriz de cadenas.')
