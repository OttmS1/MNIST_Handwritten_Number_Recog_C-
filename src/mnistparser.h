#include <boost/endian/conversion.hpp>
#include <cstdio>
#include <vector>

struct Header {
  uint32_t magicNum;
  uint32_t numImages;
  uint32_t numRows;
  uint32_t numCols;

  friend std::ostream &operator<<(std::ostream &os, Header &header) {
    os << "Magic Num: " << header.magicNum << '\n'
       << "numImages: " << header.numImages << '\n'
       << "numRows: " << header.numRows << '\n'
       << "numCols: " << header.numCols << '\n';
    return os;
  }
};

class MnistParser {
private:
  const char *m_imageFileName;
  const char *m_labelFileName;

public:
  Header header;

  MnistParser(const char *imageFileName, const char *labelFileName,
          std::vector<uint8_t> &dataVec, std::vector<uint8_t> &labelVec)
      : m_imageFileName(imageFileName), m_labelFileName(labelFileName) {
    parseImageFile(dataVec);
    parseLabelFile(labelVec);
  }

  void parseImageFile(std::vector<uint8_t> &data) {
    using namespace boost::endian;

    std::FILE *imgFile = std::fopen(m_imageFileName, "rb");
    if (imgFile == nullptr) {
      std::cerr << "Error opening file\n";
    }

    int res = std::fread(&header, sizeof(Header), 1, imgFile);
    if (res != 1) {
      std::cerr << "Error reading file header\n";
    }

    // boost endian reverse
    big_to_native_inplace(header.magicNum);
    big_to_native_inplace(header.numImages);
    big_to_native_inplace(header.numRows);
    big_to_native_inplace(header.numCols);

    const size_t size =
        (size_t)header.numImages * header.numRows * header.numCols;
    data = std::vector<uint8_t>(size, 0);

    res = std::fread(&data[0], sizeof(uint8_t), size, imgFile);
    if (res != size) {
      std::cerr << "Error reading file in data\n";
      std::cerr << "res: " << res << '\n';
    }

    std::fclose(imgFile);
  }

  void parseLabelFile(std::vector<uint8_t> &labelVec) {
    using namespace boost::endian;

    std::vector<uint32_t> labelHeader(2, 0);

    std::FILE *labelFile = std::fopen(m_labelFileName, "rb");
    if (labelFile == nullptr) {
      std::cerr << "Error opening file\n";
    }

    int res = std::fread(&labelHeader[0], sizeof(uint32_t), 2, labelFile);
    if (res != 2) {
      std::cerr << "Error reading label file header\n";
    }

    big_to_native_inplace(labelHeader[0]);
    big_to_native_inplace(labelHeader[1]);

    const size_t size = sizeof(uint8_t) * labelHeader[1];
    labelVec = std::vector<uint8_t>(size, 0);
    res = std::fread(&labelVec[0], sizeof(uint8_t), size, labelFile);
    if (res != size) {
      std::cerr << "Error reading file in labels\n";
      std::cerr << "res: " << res << '\n';
    }

    std::fclose(labelFile);
  }
};
